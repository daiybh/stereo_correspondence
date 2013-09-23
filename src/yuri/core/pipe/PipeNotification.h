/*
 * PipeNotification.h
 *
 *  Created on: 9.9.2013
 *      Author: neneko
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
								PipeNotifiable(){}
	virtual 					~PipeNotifiable() noexcept {}
	void 						notify();
	void						wait_for(duration_t dur);
private:
	mutex						var_mutex_;
	condition_variable			variable_;

};

}
}


#endif /* PIPENOTIFICATION_H_ */
