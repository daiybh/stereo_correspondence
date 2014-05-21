/*!
 * @file 		register.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		21.5.2014
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#include "OSCReceiver.h"
#include "OSCSender.h"
#include "yuri/core/Module.h"


MODULE_REGISTRATION_BEGIN("osc")
		REGISTER_IOTHREAD("osc_receiver",yuri::osc::OSCReceiver)
		REGISTER_IOTHREAD("osc_sender",yuri::osc::OSCSender)
MODULE_REGISTRATION_END()





