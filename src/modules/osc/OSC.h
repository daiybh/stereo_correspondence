/*!
 * @file 		OSC.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		21.5.2014
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef OSC_H_
#define OSC_H_
#include <string>
#include <cstdint>
#include "yuri/core/utils/make_unique.h"
#include "yuri/event/BasicEvent.h"
#include "yuri/event/EventHelpers.h"

namespace yuri {
namespace osc {

struct OSCInt {
	int32_t value;
	OSCInt(int32_t value):value(std::move(value)) {}
	template<class Iterator>
	OSCInt(Iterator& first, const Iterator& last):value(0)
	{
		if (!parse(first,last)) throw std::invalid_argument("Not enough data for int");
	}
	template<class Iterator>
	bool parse(Iterator& first, const Iterator& last)
	{
		if (std::distance(first, last)<4) return false;
		int32_t i = 0;
		uint8_t *fptr=reinterpret_cast<uint8_t*>(&i);
		ssize_t idx = 3;
		while (first != last && idx>=0) {
			fptr[idx--]=*first++;
		}
		value = i;
		return true;
	}
	std::string get_type() const {
		return "i";
	}
	event::pBasicEvent get_event() const {
		return std::make_shared<event::EventInt>(value);
	}
	std::string encode() const {
		std::string rep(4,0);
		const uint8_t *fptr=reinterpret_cast<const uint8_t*>(&value);

		for (int i=0;i<4;++i) {
			rep[i]=fptr[3-i];
		}
		return rep;
	}
};

struct OSCBool{
	bool value;
	OSCBool(bool value):value(std::move(value)) {}
	template<class Iterator>
	OSCBool(Iterator& first, const Iterator& last):value(false)
	{
		if (!parse(first,last)) throw std::invalid_argument("Not enough data for bool");
	}
	template<class Iterator>
	bool parse(Iterator& first, const Iterator& last)
	{
		return true;
	}
	std::string get_type() const {
		return value?"T":"F";
	}
	event::pBasicEvent get_event() const {
		return std::make_shared<event::EventBool>(value);
	}
	std::string encode() const {
		return {};
	}

};

struct OSCFloat {
	float value;
	OSCFloat(float value):value(std::move(value)) {}
	template<class Iterator>
	OSCFloat(Iterator& first, const Iterator& last):value(0.0f)
	{
		if (!parse(first,last)) throw std::invalid_argument("Not enough data for float");
	}
	template<class Iterator>
	bool parse(Iterator& first, const Iterator& last)
	{
		if (std::distance(first, last)<4) return false;
		float f = 0.0f;
		uint8_t *fptr=reinterpret_cast<uint8_t*>(&f);
		ssize_t idx = 3;
		while (first != last && idx>=0) {
			fptr[idx--]=*first++;
		}
		value = f;
		return true;
	}
	std::string get_type() const {
		return "f";
	}
	event::pBasicEvent get_event() const {
		return std::make_shared<event::EventDouble>(value);
	}
	std::string encode() const {
		std::string rep(4,0);
		const uint8_t *fptr=reinterpret_cast<const uint8_t*>(&value);

		for (int i=0;i<4;++i) {
			rep[i]=fptr[3-i];
		}
		return rep;
	}

};

struct OSCString {
	std::string value;
	OSCString(std::string value):value(std::move(value)) {}
	template<class Iterator>
	OSCString(Iterator& first, const Iterator& last)
	{
		if (!parse(first,last)) throw std::invalid_argument("Not enough data for string");
	}

	template<class Iterator>
	bool parse(Iterator& first, const Iterator& last)
	{
		std::string str;
		while (first!=last) {
			if (*first == 0) {
				++first;
				break;
			}
			str+=*first;
			++first;
		}
		if (size_t m = (str.size()+1)%4) {
			first=std::min(first+(4-m),last);
		}
		if (str.empty()) return false;
		value=std::move(str);
		return true;
	}
	std::string get_type() const {
		return "s";
	}
	event::pBasicEvent get_event() const {
		return std::make_shared<event::EventString>(value);
	}
	std::string encode() const {
		std::string rep = value + std::string(4-(value.size()%4),0);
		return rep;
	}
};

struct OSCTimestamp {
	OSCTimestamp(int64_t)
	{
	}
	template<class Iterator>
	OSCTimestamp(Iterator& first, const Iterator& last)
	{
		if (!parse(first,last)) throw std::invalid_argument("Not enough data for timestamp");
	}
	template<class Iterator>
	bool parse(Iterator& first, const Iterator& last)
	{
		if (std::distance(first,last)<8) {
			return false;
		}
		first+=8;
		return true;
	}
	char get_type() const {
		return 't';
	}
	event::pBasicEvent get_event() const {
		return {};//std::make_shared<event::EventString>(value);
	}
	std::string encode() const {
		std::string rep(8,0);
		return rep;
	}
};


//
//template<class Iterator>
//int32_t read_size(Iterator& first, const Iterator& last)
//{
//	if (std::distance(first,last)<4) {
//		first = last;
//	} else {
//		int32_t size = 0;
//		for (int i=0;i<4;++i) {
//			size=(size<<8)+(static_cast<int32_t>(*first)&0xFF);
//			++first;
//		}
//		return size;
//	}
//	return -1;
//}

struct OSCMidi{
	struct midi_struct {
		uint8_t port;
		uint8_t status;
		uint8_t data1;
		uint8_t data2;
	};
	midi_struct value;
	template<class Iterator>
	OSCMidi(Iterator& first, const Iterator& last)
	{
		if (!parse(first,last)) throw std::invalid_argument("Not enough data for midi");
	}
	template<class Iterator>
	bool parse(Iterator& first, const Iterator& last)
	{
		if (std::distance(first,last)<4) {
			return false;
		}
		value.port=*first++;
		value.status=*first++;
		value.data1=*first++;
		value.data2=*first++;

		return true;
	}
	std::string get_type() const {
		return "m";
	}
	event::pBasicEvent get_event() const {
		return {};//std::make_shared<event::EventString>(value);
	}
};


using event_vector = std::vector<event::pBasicEvent>;
using named_event = std::tuple<std::string, event_vector>;

template<class Iterator>
named_event parse_packet(Iterator& first, const Iterator& last, log::Log& log)
{
	event_vector events;
	try {
		OSCString name(first, last);

		log[log::verbose_debug] << "Found name " << name.value;
		if (name.value == "#bundle") {
			OSCTimestamp ts(first, last);
			while (first!=last) {
				OSCInt size(first, last);
				log[log::verbose_debug] << "Element of size " << size.value;
				auto last2 = std::min(first+size.value,last);
				std::tie(name.value, events) = parse_packet(first,last2, log);
				first = last;
			}
		} else {
			OSCString type(first,last);
			log[log::verbose_debug] << "Found types: " << type.value;
			if (type.value.size() < 2 || type.value[0]!=',') {
				first=last;
				throw std::invalid_argument("Malformed data");
			}

			for (auto it=type.value.begin()+1; it!=type.value.cend();++it) {
				switch (*it) {
					case 'f': {
						OSCFloat val(first,last);
						if (auto event = val.get_event()) {
							events.push_back(event);
						}
						log[log::verbose_debug] << "Float value: " << val.value;
					}; break;
					case 'i': {
						OSCInt val(first,last);
						if (auto event = val.get_event()) {
							events.push_back(event);
						}
						log[log::verbose_debug] << "Int value: " << val.value;
					}; break;
					case 's': {
						OSCString val(first,last);
						if (auto event = val.get_event()) {
							events.push_back(event);
						}
						log[log::verbose_debug] << "String value: " << val.value;
					}; break;
					case 'T': {
						OSCBool val(true);
						if (auto event = val.get_event()) {
							events.push_back(event);
						}
						log[log::verbose_debug] << "bool value: " << val.value;
					}; break;
					case 'F': {
						OSCBool val(false);
						if (auto event = val.get_event()) {
							events.push_back(event);
						}
						log[log::verbose_debug] << "bool value: " << val.value;
					}; break;
					case 'm': {
						OSCMidi val(first,last);
						if (auto event = val.get_event()) {
							events.push_back(event);
						}
						//events.push_back(make_shared<event::EventInt>(i));
						log[log::info] << "Midi port  " << static_cast<int>(val.value.port)
								<< ", status: " << static_cast<int>(val.value.status)
								<< ", data: " << static_cast<int>(val.value.data1)
								<< ", " << static_cast<int>(val.value.data2);
					}; break;
					default:
						{
							first=last;
							break;
						}
				}
			}
		}
		return std::make_tuple(name.value, events);
	}
	catch (std::exception&) {
		return std::make_tuple("",events);
	}
}



inline std::string encode_osc(const std::string& event_name, const event::pBasicEvent& event, bool bundle=true)
{
	std::string osc_string;
	if (bundle) {
		std::string bundle_data = encode_osc(event_name, event, false);
		osc_string = OSCString("#bundle").encode() + OSCTimestamp(0).encode() + OSCInt(bundle_data.size()).encode() + bundle_data;

	} else {
		osc_string += OSCString(event_name).encode();
		switch(event->get_type())
		{
		case event::event_type_t::integer_event: {
			OSCInt val(event::get_value<event::EventInt>(event));
			osc_string+=OSCString(","+val.get_type()).encode() + val.encode();
			}; break;
		case event::event_type_t::double_event: {
			OSCFloat val(event::get_value<event::EventDouble>(event));
			osc_string+=OSCString(","+val.get_type()).encode() + val.encode();
			}; break;
		case event::event_type_t::string_event: {
			OSCString val(event::get_value<event::EventString>(event));
			osc_string+=OSCString(","+val.get_type()).encode() + val.encode();
			}; break;
		case event::event_type_t::boolean_event: {
			OSCBool val(event::get_value<event::EventBool>(event));
			osc_string+=OSCString(","+val.get_type()).encode() + val.encode();
			}; break;


		default:
			break;
		}
	}
	return osc_string;
}






}
}


#endif /* OSC_H_ */
