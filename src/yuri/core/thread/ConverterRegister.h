/*
 * ConverterGenerator.h
 *
 *  Created on: 6.10.2013
 *      Author: neneko
 */

#ifndef CONVERTERGENERATOR_H_
#define CONVERTERGENERATOR_H_

#include "yuri/core/utils/MultiRegister.h"
#include "yuri/core/utils/Singleton.h"
#include "yuri/core/thread/ConverterThread.h"
namespace yuri {
namespace core {

/*
 * Converter generator
 * stored pair <string name, size_t cost>, where:
 *  name is name of IOThread implementing Converter interface and
 *  cost is cost of the conversion (should be greater than 0)
 * The key is pair<format_t, format_t> describing the conversion supported
 */


typedef std::pair<format_t, format_t> converter_key;
typedef std::pair<std::string, size_t> value_type;

}
}
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmismatched-tags"
#endif

namespace std {
template<>
class hash<yuri::core::converter_key> {
public:
	size_t operator()(const yuri::core::converter_key& key) const {
		return std::hash<yuri::format_t>()(key.first) ^ std::hash<yuri::format_t>()(key.second);
	}
};
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif

namespace yuri {
namespace core {



typedef utils::Singleton<generator::MultiRegister<
		converter_key,
		value_type>> 		ConverterRegister;

#ifdef YURI_MODULE_IN_TREE
#define REGISTER_CONVERTER(format1, format2, name, cost) namespace { bool reg_ ## type = yuri::core::ConverterRegister::get_instance().add_value(std::make_pair(format1, format2), std::make_pair(name, cost)); }
#else
#define REGISTER_CONVERTER(format1, format2, name, cost) /*bool iothread_reg_ ## type = */yuri::core::ConverterRegister::get_instance().add_value(std::make_pair(format1, format2), std::make_pair(name, cost));
#endif



}
}



#endif /* CONVERTERGENERATOR_H_ */
