#pragma once
#include "tensor.hpp"
#include "place_holder.hpp"
#include "variable.hpp"
#include "utils/lzw.hpp"
#include "utils/context_cast.hpp"

namespace ceras {
namespace ceras_private {
template<Tensor Tsor> inline thread_local session<Tsor>* active_session=nullptr;
template<Tensor Tsor>
struct session {
    using tensor_type=Tsor;
    using place_holder_type=place_holder<Tsor>;
    using variable_type=variable<Tsor>;
    using variable_state_type=variable_state<Tsor>;
    std::unordered_map<int,std::weak_ptr<place_holder_state<Tsor>>> place_holders_;
    std::unordered_map<int,std::weak_ptr<variable_state_type>> variables_;
    std::unordered_map<int,Tsor> forward_cache_;
    struct forward_record {Tsor input,lhs,rhs,output;};
    std::unordered_map<int,forward_record> forward_records_;
    mutable std::recursive_mutex mutex_;
    scratch_store scratch_;
    std::mt19937 rng_{random_generator};
    bool independent_;
    int phase_=1;
    explicit session(bool independent=true):independent_(independent) {}
    void seed_random(unsigned long seed) {std::lock_guard lock(mutex_);rng_.seed(seed);}
    struct execution_scope {
        session* previous=active_session<Tsor>;
        scratch_store* previous_scratch=active_scratch;
        std::mt19937* previous_rng=active_random;
        int previous_phase=learning_phase;
        session& owner;
        execution_scope(session& s):owner(s) {
            active_session<Tsor> = &s;active_scratch=&s.scratch_;
            if(s.independent_ && previous!=&s) {active_random=&s.rng_;learning_phase=s.phase_;}
        }
        ~execution_scope() {
            if(owner.independent_) owner.phase_=learning_phase;
            active_session<Tsor> = previous;active_scratch=previous_scratch;
            active_random=previous_rng;learning_phase=previous_phase;
        }
    };
    template<class F> decltype(auto) generate(F&& f) {
        std::lock_guard lock(mutex_); execution_scope scope(*this);return std::forward<F>(f)();
    }
    template<class Operation> void backward(Operation& op,Tsor const& grad) {
        std::lock_guard lock(mutex_);execution_scope scope(*this);op.backward(grad);
    }
    session(session const&)=delete;
    session& operator=(session const&)=delete;
    forward_record& forward_record_for(int id) {return forward_records_[id];}
    bool has_forward(int id) const {return forward_cache_.contains(id);}
    template<class Map> static void prune(Map& map) {
        std::erase_if(map,[](auto const& item){return item.second.expired();});
    }
    session& remember(variable_type const& v) {
        std::lock_guard lock(mutex_); prune(variables_); variables_[v.id()]=v.state_; return *this;
    }
    session& bind(place_holder_type& ph,Tsor const& value) {
        std::lock_guard lock(mutex_); prune(place_holders_);
        ph.bind(value);place_holders_[ph.id()]=ph.state_;return *this;
    }
    session& rebind(place_holder_type& ph,Tsor const& value) {return bind(ph,value);}
    void clear_forward_cache() {forward_cache_.clear();forward_records_.clear();}
    Tsor query_forward_cache(int id) const {
        auto it=forward_cache_.find(id);return it==forward_cache_.end()?Tsor{}:it->second;
    }
    void update_forward_cache(int id,Tsor value) {forward_cache_[id]=std::move(value);}
    template<class Operation> auto run(Operation& op) {
        std::lock_guard lock(mutex_);
        execution_scope scope(*this);
        std::erase_if(scratch_,[](auto const& item){return item.first.expired();});
        if constexpr(requires {Operation::is_optimizer;}) {
            try {op.forward();} catch(...) {clear_forward_cache();throw;}
            clear_forward_cache();
        } else {
            clear_forward_cache();
            try {return op.forward();} catch(...) {clear_forward_cache();throw;}
        }
    }
    template<class Operation> void tap(Operation& op) {run(op);}
    void write_original(std::ostream& out) const {
        std::lock_guard lock(mutex_);
        std::map<int,std::shared_ptr<variable_state_type>> live;
        for(auto const& [id,weak]:variables_) if(auto state=weak.lock()) live.emplace(id,state);
        for(auto const& [id,state]:live) out<<id<<' ';
        out<<'\n';
        for(auto const& [id,state]:live) write_tensor(out,state->data_);
        if(!out) throw std::runtime_error("Cannot write session");
    }
    void read_original(std::istream& in) {
        std::lock_guard lock(mutex_);
        std::string line;
        char c;
        while(in.get(c) && c!='\n') {
            if(line.size()>=tensor_io_limits.max_line_bytes) throw std::runtime_error("Session header too long");
            line+=c;
        }
        if(!in)
            throw std::runtime_error("Invalid session header");
        std::istringstream ids(line);
        std::map<int,std::pair<std::shared_ptr<variable_state_type>,Tsor>> staged;
        int id;
        std::size_t staged_bytes=0;
        while(ids>>id) {
            auto it=variables_.find(id);
            auto state=it==variables_.end()?nullptr:it->second.lock();
            if(!state || staged.contains(id)) throw std::runtime_error("Unknown or duplicate session variable");
            Tsor data;read_tensor(in,data);
            if(in.fail() || data.shape()!=state->data_.shape()) throw std::runtime_error("Invalid session tensor");
            staged_bytes=checked_add(staged_bytes,checked_multiply(data.size(),sizeof(typename Tsor::value_type)));
            if(staged_bytes>tensor_io_limits.max_bytes) throw std::length_error("session byte budget");
            staged.emplace(id,std::make_pair(state,std::move(data)));
        }
        if(!ids.eof()) throw std::runtime_error("Invalid session ID");
        in>>std::ws;if(!in.eof()) throw std::runtime_error("Trailing session data");
        // All validation and allocations precede mutation; shared owners keep targets live.
        for(auto& [key,item]:staged) std::copy(item.second.begin(),item.second.end(),item.first->data_.begin());
        clear_forward_cache();
    }
    void save_original(std::string const& path) const {
        std::ofstream out(path);write_original(out);
    }
    void restore_original(std::string const& path) {
        std::ifstream in(path);if(!in) throw std::runtime_error("Cannot open session: "+path);
        read_original(in);
    }
    void save(std::string const& path) const {
        std::ostringstream text;write_original(text);
        std::istringstream in(text.str()); std::ofstream out(path,std::ios::binary);
        if(!out) throw std::runtime_error("Cannot save session: "+path);
        lzw::compress(in,out);if(!out) throw std::runtime_error("Session write failed");
    }
    void restore(std::string const& path) {
        std::ifstream in(path,std::ios::binary);if(!in) throw std::runtime_error("Cannot open session: "+path);
        std::ostringstream text;
        if(lzw::decompress(in,text,tensor_io_limits.max_bytes)!=0) throw std::runtime_error("Invalid compressed session");
        std::istringstream parsed(text.str());read_original(parsed);
    }
    void serialize(std::string const& path) const {save(path);}
    void deserialize(std::string const& path) {restore(path);}
};
}
template<Tensor Tsor> ceras_private::session<Tsor>& get_default_session() {
    if(ceras_private::active_session<Tsor>) return *ceras_private::active_session<Tsor>;
    static thread_local ceras_private::session<Tsor> instance(false);
    return instance;
}
template<Tensor Tsor> auto& bind(place_holder<Tsor>& ph,Tsor const& value) {
    auto& s=get_default_session<Tsor>();s.bind(ph,value);return s;
}
template<class Op> auto run(Op& op) {
    auto& s=get_default_session<typename Op::tensor_type>();return s.run(op);
}
}
