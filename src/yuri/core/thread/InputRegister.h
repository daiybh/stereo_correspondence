/*!
 * @file 		InputRegister.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		01.03.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef INPUTREGISTER_H_
#define INPUTREGISTER_H_

#include "yuri/core/utils/SingleRegister.h"
#include "yuri/core/utils/Singleton.h"
#include "yuri/core/thread/InputThread.h"

namespace yuri {
namespace core {

using input_generator = std::function<std::vector<core::InputDeviceInfo>(void)>;
using InputRegister = yuri::utils::Singleton<generator::SingleRegister<
		std::string,
		input_generator>>;

#ifdef YURI_MODULE_IN_TREE
#define REGISTER_INPUT_THREAD(name, enumerate) namespace { bool reg_ ## type = yuri::core::InputRegister::get_instance().add_value(name, enumerate); }
#else
#define REGISTER_INPUT_THREAD(name, enumerate) /*bool iothread_reg_ ## type = */yuri::core::InputRegister::get_instance().add_value(name, enumerate);
#endif



}
}

SINGLETON_DECLARE_HELPER(yuri::core::InputRegister)


#endif /* INPUTREGISTER_H_ */
