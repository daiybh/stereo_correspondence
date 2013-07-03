/*
 * BasicEvent.h
 *
 *  Created on: 2.7.2013
 *      Author: neneko
 */

#ifndef BASICEVENT_H_
#define BASICEVENT_H_
#include "yuri/core/types.h"
#include <initializer_list>
namespace yuri{
namespace core {

enum class event_type_t {
	bang_event,
	boolean_event,
	integer_event,
	double_event,
	time_event,
	string_event,

	vector_event,
	dictionary_event,
};
class BasicEvent;
typedef std::shared_ptr<BasicEvent> pBasicEvent;
class BasicEvent: public std::enable_shared_from_this<BasicEvent> {
public:
						BasicEvent() = delete;
						BasicEvent(event_type_t type):type_(type),
								timestamp_(std::chrono::steady_clock::now()) {}
						BasicEvent(const BasicEvent&) = delete;
						BasicEvent(BasicEvent&&) = delete;
	void 				operator=(const BasicEvent&) = delete;
	void 				operator=(BasicEvent&&) = delete;

	virtual 			~BasicEvent() {}

	event_type_t 		get_type() const
							{ return type_; }
	time_value 			get_timestamp() const
							{ return timestamp_; }

private:
	event_type_t		type_;
	const time_value	timestamp_;

};


template<event_type_t type, class Value>
class EventBase: public BasicEvent {
public:
	typedef Value 		stored_type;
	EventBase(const Value& value): BasicEvent(type),value_(value) {}
	EventBase(Value&& value): BasicEvent(type),value_(std::move(value)) {}
	EventBase&			operator=(const Value& value)
							{ value_ = value; return &this;}
	EventBase&			operator=(Value&& value)
							{ value_ = std::move(value); return &this;}
						operator Value() { return value_; }
						operator Value() const { return value_; }

	Value&				get_value()	{ return value_; }
	const Value&		get_value() const { return value_; }
private:
	Value 				value_;
};

template<event_type_t type, class Value>
class EventBaseRanged: public EventBase<type, Value> {
public:
	EventBaseRanged(const Value& value, const Value& min_value=std::numeric_limits<Value>::min(), const Value& max_value=std::numeric_limits<Value>::max()):
		EventBase<type, Value>(value),min_(min_value),max_(max_value) {}
	EventBaseRanged(Value&& value, const Value& min_value=std::numeric_limits<Value>::min(), const Value& max_value=std::numeric_limits<Value>::max()):
		EventBase<type, Value>(std::move(value)),min_(min_value),max_(max_value)  {}

	const Value&		get_min_value() const { return min_; }
	const Value&		get_max_value() const { return max_; }
	bool				range_min_specified() const { return min_ != std::numeric_limits<Value>::min(); }
	bool				range_max_specified() const { return max_ != std::numeric_limits<Value>::max(); }
	bool				range_specified() const { return range_min_specified() && range_max_specified(); }
private:
	const Value			min_;
	const Value			max_;
};

// Specialization for Bang
template<>
class EventBase<event_type_t::bang_event, void>: public BasicEvent
{
public:
	typedef void 		stored_type;
	EventBase():BasicEvent(event_type_t::bang_event) {}

};



class EventVector: public EventBase<event_type_t::vector_event, std::vector<pBasicEvent> >
{
public:
	EventVector(const std::vector<pBasicEvent>& value):EventBase<event_type_t::vector_event, std::vector<pBasicEvent>>(value)
		{}
	EventVector(std::vector<pBasicEvent>&& value):EventBase<event_type_t::vector_event, std::vector<pBasicEvent>>(std::move(value))
		{}
	EventVector(std::initializer_list<pBasicEvent> list):EventBase<event_type_t::vector_event, std::vector<pBasicEvent>>(list)
		{}

	// Passing through vector interface
	typedef stored_type::value_type 		value_type;
	typedef stored_type::allocator_type 	allocator_type;
	typedef stored_type::size_type 			size_type;
	typedef stored_type::difference_type	difference_type;
	typedef stored_type::reference 			reference;
	typedef stored_type::const_reference 	const_reference;
	typedef stored_type::pointer 			pointer;
	typedef stored_type::const_pointer	 	const_pointer;
	typedef stored_type::iterator 			iterator;
	typedef stored_type::const_iterator 	const_iterator;
	typedef stored_type::reverse_iterator 	reverse_iterator;
	typedef stored_type::const_reverse_iterator
											const_reverse_iterator;

	bool 									empty() const { return get_value().empty(); }
	size_type 								size() const { return get_value().size(); }
	size_type 								max_size() const { return get_value().max_size(); }
	size_type 								capacity() const { return get_value().capacity(); }
	void									reserve(size_type size) { get_value().reserve(size); }
	void									shrink_to_fit() {  get_value().shrink_to_fit(); }
	reference       						at( size_type pos ) { return get_value().at(pos); }
	const_reference 						at( size_type pos ) const { return get_value().at(pos); }
	reference       						operator[]( size_type pos ) { return get_value()[pos]; }
	const_reference 						operator[]( size_type pos ) const { return get_value()[pos]; }
	reference 								front() { return get_value().front(); }
	const_reference 						front() const { return get_value().front(); }
	reference 								back() { return get_value().back(); }
	const_reference 						back() const { return get_value().back(); }
	pointer 								data() { return get_value().data(); }
	const_pointer 							data() const { return get_value().data(); }
	iterator 								begin() { return get_value().begin(); }
	const_iterator 							begin() const { return get_value().begin(); }
	const_iterator 							cbegin() const { return get_value().cbegin(); }
	iterator 								end() { return get_value().end(); }
	const_iterator 							end() const { return get_value().end(); }
	const_iterator 							cend() const { return get_value().cend(); }
	reverse_iterator 						rbegin() { return get_value().rbegin(); }
	const_reverse_iterator 					rbegin() const { return get_value().rbegin(); }
	const_reverse_iterator 					crbegin() const { return get_value().crbegin(); }
	reverse_iterator 						rend() { return get_value().rend(); }
	const_reverse_iterator 					rend() const { return get_value().rend(); }
	const_reverse_iterator 					crend() const { return get_value().crend(); }

};
class EventDict: public EventBase<event_type_t::dictionary_event, std::map<std::string, pBasicEvent> >
{
	public:
	EventDict(const std::map<std::string, pBasicEvent>& value):EventBase<event_type_t::dictionary_event, std::map<std::string, pBasicEvent>>(value)
		{}
	EventDict(std::map<std::string, pBasicEvent>&& value):EventBase<event_type_t::dictionary_event, std::map<std::string, pBasicEvent>>(std::move(value))
		{}
	EventDict(std::initializer_list<stored_type::value_type> list):EventBase<event_type_t::dictionary_event, stored_type>(list)
		{}
};
typedef EventBase<event_type_t::bang_event, void> EventBang;
typedef EventBase<event_type_t::boolean_event, bool> EventBool;
typedef EventBaseRanged<event_type_t::integer_event, int64_t> EventInt;
typedef EventBaseRanged<event_type_t::double_event, long double> EventDouble;
typedef EventBase<event_type_t::string_event, std::string> EventString;
typedef EventBase<event_type_t::time_event, time_value> EventTime;


}

}


#endif /* BASICEVENT_H_ */
