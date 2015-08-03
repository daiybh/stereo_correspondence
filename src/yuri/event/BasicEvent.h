/*!
 * @file 		BasicEvent.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		2.07.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef BASICEVENT_H_
#define BASICEVENT_H_
#include "yuri/core/utils/new_types.h"
#include "yuri/core/utils/time_types.h"
#include <initializer_list>
#include <vector>
#include <map>
namespace yuri{

namespace event {

enum class event_type_t {
								undetermined_event,
								bang_event,
								boolean_event,
								integer_event,
								double_event,
								time_event,
								string_event,
								duration_event,

								vector_event,
								dictionary_event,
};
struct bang_t {};

class BasicEvent;
using pBasicEvent = std::shared_ptr<BasicEvent>;

class BasicEvent: 	public std::enable_shared_from_this<BasicEvent> {
public:
	EXPORT 						BasicEvent() = delete;
	EXPORT						BasicEvent(event_type_t type):type_(type),
											timestamp_() {}
	EXPORT 						BasicEvent(const BasicEvent&) = delete;
	EXPORT 						BasicEvent(BasicEvent&&) = delete;
	EXPORT void 				operator=(const BasicEvent&) = delete;
	EXPORT void 				operator=(BasicEvent&&) = delete;

	EXPORT virtual 				~BasicEvent() {}

	event_type_t 		get_type() const
									{ return type_; }
	timestamp_t			get_timestamp() const
									{ return timestamp_; }
	pBasicEvent					get_copy() const
									{ return do_get_copy(); }

private:
	virtual pBasicEvent	do_get_copy() const = 0;
	event_type_t				type_;
	const timestamp_t			timestamp_;
};


template<event_type_t type, class Value>
class EventBase: public BasicEvent {
public:
	typedef Value 				stored_type;
								EventBase()
									:BasicEvent(type) {}
								EventBase(const Value& value)
									:BasicEvent(type),value_(value) {}
								EventBase(Value&& value)
									:BasicEvent(type),value_(std::move(value)) {}
	virtual 					~EventBase() {}
	EventBase&					operator=(const Value& value)
									{ value_ = value; return &this;}
	EventBase&					operator=(Value&& value)
									{ value_ = std::move(value); return &this;}
								operator Value()
									{ return value_; }
								operator Value() const
									{ return value_; }

	Value&						get_value()
									{ return value_; }
	const Value&				get_value() const
									{ return value_; }
private:
	Value 						value_;
	virtual pBasicEvent 		do_get_copy() const
									{ return std::make_shared<EventBase<type, Value>>(value_); }
};

template<event_type_t type, class Value>
class EventBaseRanged: public EventBase<type, Value> {
public:
								EventBaseRanged(const Value& value,
												const Value& min_value=std::numeric_limits<Value>::min(),
												const Value& max_value=std::numeric_limits<Value>::max()):
									EventBase<type, Value>(value),min_(min_value),max_(max_value) {}

								EventBaseRanged(Value&& value,
												const Value& min_value=std::numeric_limits<Value>::min(),
												const Value& max_value=std::numeric_limits<Value>::max()):
									EventBase<type, Value>(std::move(value)),min_(min_value),max_(max_value)  {}
	virtual						~EventBaseRanged() {}
	const Value&				get_min_value() const
									{ return min_; }
	const Value&				get_max_value() const
									{ return max_; }
	bool						range_min_specified() const
									{ return min_ != std::numeric_limits<Value>::min(); }
	bool						range_max_specified() const
									{ return max_ != std::numeric_limits<Value>::max(); }
	bool						range_specified() const
									{ return range_min_specified() && range_max_specified(); }
private:
	const Value					min_;
	const Value					max_;
	virtual pBasicEvent 		do_get_copy() const
									{ return std::make_shared<EventBaseRanged<type, Value>>(this->get_value(), min_, max_); }
};

// Specialization for Bang
template<>
class EventBase<event_type_t::bang_event, bang_t>: public BasicEvent
{
public:
	typedef bang_t 				stored_type;
								EventBase()
									:BasicEvent(event_type_t::bang_event) {}
	bang_t						get_value() const { return bang_t(); }
private:
	virtual pBasicEvent 		do_get_copy() const
									{ return std::make_shared<EventBase<event_type_t::bang_event, bang_t>>(); }

};



class EventVector: public EventBase<event_type_t::vector_event, std::vector<pBasicEvent> >
{
public:
								EventVector()
									:EventBase<event_type_t::vector_event, std::vector<pBasicEvent>>()
									{}
								EventVector(const std::vector<pBasicEvent>& value)
									:EventBase<event_type_t::vector_event, std::vector<pBasicEvent>>(value)
									{}
								EventVector(std::vector<pBasicEvent>&& value)
									:EventBase<event_type_t::vector_event, std::vector<pBasicEvent>>(std::move(value))
									{}
								EventVector(std::initializer_list<pBasicEvent> list)
									:EventBase<event_type_t::vector_event, std::vector<pBasicEvent>>(list)
									{}
								template<class... Args>
								EventVector(const pBasicEvent& ev0, Args... args):
									EventBase<event_type_t::vector_event, std::vector<pBasicEvent> >({ev0, args...})
									{}
	virtual 					~EventVector() {}
	// Passing through vector interface
	typedef stored_type::value_type
								value_type;
	typedef stored_type::allocator_type
								allocator_type;
	typedef stored_type::size_type
								size_type;
	typedef stored_type::difference_type
								difference_type;
	typedef stored_type::reference
								reference;
	typedef stored_type::const_reference
								const_reference;
	typedef stored_type::pointer
								pointer;
	typedef stored_type::const_pointer
								const_pointer;
	typedef stored_type::iterator
								iterator;
	typedef stored_type::const_iterator
								const_iterator;
	typedef stored_type::reverse_iterator
								reverse_iterator;
	typedef stored_type::const_reverse_iterator
								const_reverse_iterator;

	bool 						empty() const
									{ return get_value().empty(); }
	size_type 					size() const
									{ return get_value().size(); }
	size_type 					max_size() const
									{ return get_value().max_size(); }
	size_type 					capacity() const
									{ return get_value().capacity(); }
	void						reserve(size_type size)
									{ get_value().reserve(size); }
	void						shrink_to_fit()
									{  get_value().shrink_to_fit(); }
	reference       			at( size_type pos )
									{ return get_value().at(pos); }
	const_reference 			at( size_type pos ) const
									{ return get_value().at(pos); }
	reference       			operator[]( size_type pos )
									{ return get_value()[pos]; }
	const_reference 			operator[]( size_type pos ) const
									{ return get_value()[pos]; }
	reference 					front()
									{ return get_value().front(); }
	const_reference 			front() const
									{ return get_value().front(); }
	reference 					back()
									{ return get_value().back(); }
	const_reference 			back() const
									{ return get_value().back(); }
	pointer 					data()
									{ return get_value().data(); }
	const_pointer 				data() const
									{ return get_value().data(); }
	iterator 					begin()
									{ return get_value().begin(); }
	const_iterator 				begin() const
									{ return get_value().begin(); }
	const_iterator 				cbegin() const
									{ return get_value().cbegin(); }
	iterator 					end()
									{ return get_value().end(); }
	const_iterator 				end() const
									{ return get_value().end(); }
	const_iterator 				cend() const
									{ return get_value().cend(); }
	reverse_iterator 			rbegin()
									{ return get_value().rbegin(); }
	const_reverse_iterator 		rbegin() const
									{ return get_value().rbegin(); }
	const_reverse_iterator 		crbegin() const
									{ return get_value().crbegin(); }
	reverse_iterator 			rend()
									{ return get_value().rend(); }
	const_reverse_iterator 		rend() const
									{ return get_value().rend(); }
	const_reverse_iterator 		crend() const
									{ return get_value().crend(); }

	void						push_back(const value_type& value)
									{ get_value().push_back(value); }
	void						push_back(value_type&& value)
									{ get_value().push_back(value); }
private:
	virtual pBasicEvent			do_get_copy() const {
									stored_type tmp_vec {};
									for (const auto& event: *this) tmp_vec.push_back(event->get_copy());
									return std::make_shared<EventVector>(tmp_vec);
								}
};
class EventDict: public EventBase<event_type_t::dictionary_event, std::map<std::string, pBasicEvent> >
{
public:
								EventDict()
									:EventBase<event_type_t::dictionary_event, std::map<std::string, pBasicEvent>>()
									{}
								EventDict(const std::map<std::string, pBasicEvent>& value)
									:EventBase<event_type_t::dictionary_event, std::map<std::string, pBasicEvent>>(value)
									{}
								EventDict(std::map<std::string, pBasicEvent>&& value)
									:EventBase<event_type_t::dictionary_event, std::map<std::string, pBasicEvent>>(std::move(value))
									{}
								EventDict(std::initializer_list<stored_type::value_type> list)
									:EventBase<event_type_t::dictionary_event, stored_type>(list)
									{}
								template<class... Args>
								EventDict(const std::pair<const std::string, pBasicEvent>& ev0, Args... args):
									EventBase<event_type_t::dictionary_event, stored_type >({ev0, args...})
									{}
private:
virtual pBasicEvent				do_get_copy() const {
									stored_type tmp_map {};
									for (const auto& event: get_value()) tmp_map.insert(std::make_pair(event.first, event.second->get_copy()));
									return std::make_shared<EventDict>(tmp_map);
								}
};


typedef EventBase<event_type_t::bang_event, bang_t>
								EventBang;
typedef EventBase<event_type_t::boolean_event, bool>
								EventBool;
typedef EventBaseRanged<event_type_t::integer_event, int64_t>
								EventInt;
typedef EventBaseRanged<event_type_t::double_event, long double>
								EventDouble;
typedef EventBase<event_type_t::string_event, std::string>
								EventString;
typedef EventBase<event_type_t::time_event, timestamp_t>
								EventTime;
typedef EventBase<event_type_t::duration_event, duration_t>
								EventDuration;


template<class EventType>
struct event_traits {
};

template<>
struct event_traits<EventBang> {
	typedef bang_t stored_type;
	static event_type_t event_type() { return event_type_t::bang_event; }
};

template<>
struct event_traits<EventBool> {
	typedef bool stored_type;
	static event_type_t event_type() { return event_type_t::boolean_event; }
};
template<>
struct event_traits<EventInt> {
	typedef int64_t stored_type;
	static event_type_t event_type() { return event_type_t::integer_event; }
};
template<>
struct event_traits<EventDouble> {
	typedef long double stored_type;
	static event_type_t event_type() { return event_type_t::double_event; }
};
template<>
struct event_traits<EventString> {
	typedef std::string stored_type;
	static event_type_t event_type() { return event_type_t::string_event; }
};
template<>
struct event_traits<EventTime> {
	typedef timestamp_t stored_type;
	static event_type_t event_type() { return event_type_t::time_event; }
};
template<>
struct event_traits<EventDuration> {
	typedef duration_t stored_type;
	static event_type_t event_type() { return event_type_t::duration_event; }
};
template<>
struct event_traits<EventVector> {
	typedef std::vector<pBasicEvent> stored_type;
	static event_type_t event_type() { return event_type_t::vector_event; }
};
template<>
struct event_traits<EventDict> {
	typedef std::map<std::string, pBasicEvent> stored_type;
	static event_type_t event_type() { return event_type_t::dictionary_event; }
};

}

}


#endif /* BASICEVENT_H_ */
