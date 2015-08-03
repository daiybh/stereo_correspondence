/*!
 * @file 		PipeNotification.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		9.9.2013
 * @date		1.12.2014
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef PIPENOTIFICATION_H_
#define PIPENOTIFICATION_H_
#include "yuri/core/utils/new_types.h"
#include "yuri/core/utils/time_types.h"
#include <condition_variable>

namespace yuri {
namespace core {

using pPipeNotifiable = std::shared_ptr<class PipeNotifiable>;
using pwPipeNotifiable = std::weak_ptr<class PipeNotifiable>;
class PipeNotifiable {
public:
	EXPORT 						PipeNotifiable():pending_notification_{false}{}
	EXPORT virtual 				~PipeNotifiable() noexcept {}
	EXPORT void 				notify();
	EXPORT void					wait_for(duration_t dur);
private:
	yuri::mutex					var_mutex_;
	std::condition_variable		variable_;
	bool						pending_notification_;

};

}
}


#endif /* PIPENOTIFICATION_H_ */
