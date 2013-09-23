/*
 * Parameter.h
 *
 *  Created on: 11.9.2013
 *      Author: neneko
 */

#ifndef PARAMETER_H_
#define PARAMETER_H_

#include "yuri/core/utils/new_types.h"
#include "yuri/event/BasicEvent.h"
#include "yuri/event/EventHelpers.h"
#include "param_helpers.h"

namespace yuri {
namespace core {





class EXPORT Parameter {
public:
								Parameter(const std::string& name, event::pBasicEvent value= event::pBasicEvent(), const std::string& description=std::string()):
									name_(name),description_(description),value_(value) {}
	template<typename T>
								Parameter(const std::string& name, const T& value, const std::string& description=std::string()):
									name_(name),description_(description),value_(detail::suitable_event_type<T>::create_new(value)) {}
								~Parameter() noexcept = default;
								Parameter(const Parameter&) = default;
								Parameter(Parameter&&) = default;
	Parameter&					operator=(const Parameter&) = default;
	Parameter&					operator=(Parameter&&) = default;
	Parameter&					operator=(const event::pBasicEvent& value) { set_value(value); return *this; }

	template<typename T>
	Parameter&					operator=(const T& value) { value_ = detail::suitable_event_type<T>::create_new(value); return *this; }


	const std::string&			get_name() const { return name_; }
	const event::pBasicEvent&	get_value() const { return value_; }
	const std::string&			get_description() const { return description_; }

	void						set_value(event::pBasicEvent value) { value_ = value; }
	Parameter&					operator[](const std::string& desc){description_=desc; return *this;}

	template<typename T>
	T							get() const;
	template<typename T>
	T							get_indexed(index_t index) const;
	template<typename T>
	T							get_indexed(const std::string& index) const;

private:
	std::string 				name_;
	std::string					description_;
	event::pBasicEvent			value_;
};



template<typename T>
T Parameter::get() const {
	return event::lex_cast_value<T>(value_);
}

template<typename T>
T Parameter::get_indexed(index_t index) const {
	using namespace yuri::event;
	const event_type_t type = value_->get_type();
	if (type != event_type_t::vector_event) throw std::invalid_argument("Can't index non-vector values");
	const shared_ptr<EventVector> vec_value = dynamic_pointer_cast<EventVector>(value_);
	return event::lex_cast_value<T>(vec_value->at(index));
}

template<typename T>
T Parameter::get_indexed(const std::string& index) const {
	using namespace yuri::event;
	const event_type_t type = value_->get_type();
	if (type != event_type_t::dictionary_event) throw std::invalid_argument("Can't string index non-dictionary values");
	const shared_ptr<EventDict> dict_value = dynamic_pointer_cast<EventDict>(value_);
	// TODO: OK, this is ugly, but EventDict still doesn't pass the map interface...
	const auto& tmp_val = dict_value->get_value();
	auto it = tmp_val.find(index);
	if (it == tmp_val.end()) return std::range_error("No index " + index + " in the value");
	return event::lex_cast_value<T>(*it);
}




}
}
#endif /* PARAMETER_H_ */
