/*!
 *
 * The file contains static tests for the StateTransitionTable.h
 *
 * @file 		test_state_table.h
 * @author 		Anastasia Kuznetsova <kuzneana@gmail.com>
 * @date 		21.2.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "tests/catch.hpp"
#include "yuri/core/utils/StateTransitionTable.h"
#include <sstream>

namespace yuri{

	namespace{

	enum class test_state{
        first_state = 0,
        second_state,
        third_state
    };

    enum class test_event{
        first_event = 0,
        second_event,
        third_event
    };


    Timer t;

    class TestObject : public StateTransitionTable<test_event, test_state>{
		public:
            TestObject(log::Log& l) : StateTransitionTable(l, test_state::first_state), first_(2), second_(2), result_(0) {
                define_transition_table();
            }

            ~TestObject() noexcept{}

            void define_transition_table() override{
                add_transition(test_state::first_state, test_event::first_event,
                               test_state::second_state, std::bind(&TestObject::do_nothing, this));
                add_transition(test_state::first_state, test_event::first_event,
                               test_state::second_state, std::bind(&TestObject::multiply_parameters, this));
                add_transition(test_state::second_state, test_event::second_event,
                               test_state::third_state, std::bind(&TestObject::do_default_action, this));
			}

            void multiply_parameters(){
                result_ = first_*second_;
            }

            void do_default_action() override{
                first_=1;
                second_=1;
                result_= first_+second_;
            }

            void do_nothing(){}

            int first_;
            int second_;
            int result_;
		};


        /**
         * The function verifies maintenance of deterministic state in the transition table.
         * @brief test_deterministic_state
         */
        void test_deterministic_state(TestObject& test_obj){
           REQUIRE( test_obj.states_.size()==2 );
        }

        /**
         * The function tests functionality of function pointers.
         * @brief test_call_function
         */
        void test_call_function(TestObject& test_obj){
            //std::cout<<t.get_duration().value <<"  "<<t.get_duration().value/1000 <<std::endl;
            test_obj.process_event(test_event::first_event);
            REQUIRE( test_obj.result_==(test_obj.first_*test_obj.second_) );
        }

        void test_transition_among_states(TestObject& test_obj){
            REQUIRE( test_obj.curr_state_==test_state::second_state);
            test_obj.process_event(test_event::second_event);
            REQUIRE( test_obj.curr_state_==test_state::third_state);
        }

        /**
         * The function tests functionality of function pointers, calling default function.
         * @brief test_call_function
         */
        void test_call_default_function(TestObject& test_obj){
            REQUIRE( test_obj.curr_state_ == test_state::third_state);
            REQUIRE( test_obj.result_==(test_obj.first_+test_obj.second_) );
        }

	}

    TEST_CASE("Test state transition table", "[module]"){
    	std::stringstream ss;
		log::Log l(ss);
		auto test_obj = std::make_shared<TestObject> (l);

        test_deterministic_state(*test_obj);
        test_call_function(*test_obj);
        test_transition_among_states(*test_obj);
        test_call_default_function(*test_obj);
    }


}
