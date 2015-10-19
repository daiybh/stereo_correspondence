/*!
 * @file 		StateTransitionTable.h
 * @author 		Anastasia Kuznetsova <kuzneana@gmail.com>
 * @date 		20. 4. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef STATETRANSITIONTABLE_H_
#define STATETRANSITIONTABLE_H_

#include "yuri/core/Module.h"
#include <functional>
#include <unordered_map>

namespace yuri{

template <class event_t,
          class state_t>
class StateTransitionTable{
public:

    StateTransitionTable() = delete;
    StateTransitionTable(log::Log &log, state_t curr_state):
    	log_(log), curr_state_(curr_state) {}
    virtual ~StateTransitionTable() noexcept {}

    virtual void define_transition_table() = 0;

    void process_event(const event_t &event)
    {
        auto key = Key{curr_state_, event};
        auto search = states_.find(key);
        if (search != states_.end()){
            search->second.action_();
            curr_state_ = search->second.target_state_;
        } else {
            do_default_action();
        }
    }

    /**
     * This method will be called if action for the transition is not defined
     * @brief do_default_action
     */
    virtual void do_default_action() {}

    /**
     * Define the initial type for the function pointer
     */
    using action_t =  std::function<void()>;

    struct Key
    {
      Key(state_t from_state, event_t event) : from_state_(from_state), event_(event) {}

      bool operator==(const Key &other) const{
          return from_state_ == other.from_state_ && event_ == other.event_;
      }

      state_t from_state_;
      event_t event_;
    };

    struct KeyHasher
    {

      std::size_t operator()(const Key& key) const
      {
    	  return (hash_helper(key.from_state_)
    			  ^ (hash_helper(key.event_) << 1)) >> 1;
      }
    private:
      	template<class T>
		typename std::enable_if<std::is_enum<T>::value, std::size_t>::type
		static hash_helper(const T& val) {
      		using std::hash;
      		using under_type = typename std::underlying_type<T>::type;
      		return hash<under_type>()(static_cast<under_type>(val));
		}
      	template<class T>
		typename std::enable_if<!std::is_enum<T>::value, std::size_t>::type
		static hash_helper(const T& val) {
      		using std::hash;
      		return hash<T>()(val);
		}
    };

    struct Transition{
        Transition() {}
        Transition(state_t target_state, action_t action) : target_state_(target_state), action_(action) {}
        
        state_t target_state_;
        action_t action_;
    };

    void add_transition(const state_t& from_state,const  event_t& event,
                        const state_t& to_state, const action_t action)
    {
    	states_[Key(from_state, event)]=Transition(to_state, action);
    }


    log::Log& log_;
    state_t curr_state_;
    std::unordered_map<Key,Transition, KeyHasher> states_;
};




}

#endif
