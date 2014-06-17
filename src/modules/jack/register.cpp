/*!
 * @file 		register.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		28.5.2014
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#include "JackInput.h"
#include "JackOutput.h"
#include "yuri/core/Module.h"

MODULE_REGISTRATION_BEGIN("jack")
		REGISTER_IOTHREAD("jack_output",yuri::jack::JackOutput)
		REGISTER_IOTHREAD("jack_input",yuri::jack::JackInput)
MODULE_REGISTRATION_END()






