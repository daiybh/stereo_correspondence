/*
 * forward.h
 *
 *  Created on: 22.2.2013
 *      Author: neneko
 */

#ifndef FORWARD_H_
#define FORWARD_H_

#include "types.h"

namespace yuri {

namespace core
{
 	 class ThreadChild;
 	 typedef yuri::shared_ptr<ThreadChild> pThreadChild;
 	 class ThreadBase;
 	 typedef yuri::weak_ptr<class ThreadBase> pwThreadBase;
 	 typedef yuri::shared_ptr<class ThreadBase> pThreadBase;

 	 class BasicFrame;
 	 typedef yuri::shared_ptr<class BasicFrame> pBasicFrame;

 	 class Parameters;
 	 typedef yuri::shared_ptr<class Parameters> pParameters;

 	 class Parameter;
	 typedef yuri::shared_ptr<class Parameter> pParameter;

 	 class BasicPipe;
 	 typedef yuri::shared_ptr<class BasicPipe> pBasicPipe;

 	 class BasicIOThread;
 	 typedef yuri::shared_ptr<class BasicIOThread> pBasicIOThread;

}


}


#endif /* FORWARD_H_ */
