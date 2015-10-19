/*!
 * @file 		register.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		19. 10. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#include "DelayEstimation.h"
#include "TimestampObserver.h"
#include "OnepcProtocolCohort.h"
#include "OnepcProtocolCoordinator.h"
#include "TwopcProtocolCohort.h"
#include "TwopcProtocolCoordinator.h"
#include "TwopcTimeoutProtocolCohort.h"
#include "TwopcTimeoutProtocolCoordinator.h"
#include "PlaybackController.h"
namespace yuri {
namespace synchronization {

MODULE_REGISTRATION_BEGIN("synchronization")
		REGISTER_IOTHREAD("delay_estimation", DelayEstimation)
		REGISTER_IOTHREAD("onepc_protocol_cohort", OnepcProtocolCohort)
		REGISTER_IOTHREAD("onepc_protocol_coordinator", OnepcProtocolCoordinator)
		REGISTER_IOTHREAD("playback_controller", PlaybackController)
		REGISTER_IOTHREAD("timestamp_observer",TimestampObserver)

		REGISTER_IOTHREAD("twopc_protocol_cohort",                  TwopcProtocolCohort)
		REGISTER_IOTHREAD("twopc_protocol_coordinator",             TwopcProtocolCoordinator)
		REGISTER_IOTHREAD("twopc_timeout_protocol_cohort",          TwopcTimeoutProtocolCohort)
		REGISTER_IOTHREAD("twopc_timeout_protocol_coordinator",     TwopcTimeoutProtocolCoordinator)
MODULE_REGISTRATION_END()

}
}


