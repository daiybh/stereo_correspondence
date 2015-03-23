/*!
 * @file 		register.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		22.03.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "yuri/core/thread/IOThreadGenerator.h"
#include "SineGenerator.h"


namespace yuri {
namespace audio_gen {


MODULE_REGISTRATION_BEGIN("audio_gen")
		REGISTER_IOTHREAD("sine_generator",SineGenerator)
MODULE_REGISTRATION_END()

}
}


