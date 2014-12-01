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

using pPipeNotifiable = shared_ptr<class PipeNotifiable>;
using pwPipeNotifiable = weak_ptr<class PipeNotifiable>;
class PipeNotifiable {
public:
								PipeNotifiable():pending_notification_{false}{}
	virtual 					~PipeNotifiable() noexcept {}
	void 						notify();
	void						wait_for(duration_t dur);
private:
	mutex						var_mutex_;
	condition_variable			variable_;
	bool						pending_notification_;

};

}
}


#endif /* PIPENOTIFICATION_H_ */
