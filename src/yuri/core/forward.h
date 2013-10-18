/*
 * forward.h
 *
 *  Created on: 22.2.2013
 *      Author: neneko
 */

#ifndef FORWARD_H_
#define FORWARD_H_

#include "yuri/core/utils/new_types.h"

namespace yuri {

namespace core
{
 	 class ThreadChild;
 	 typedef yuri::shared_ptr<ThreadChild> pThreadChild;
 	 class ThreadBase;
 	 typedef yuri::weak_ptr<class ThreadBase> pwThreadBase;
 	 typedef yuri::shared_ptr<class ThreadBase> pThreadBase;
 	 typedef yuri::shared_ptr<class ThreadBase const> pcThreadBase;

 	 class Frame;
 	 typedef yuri::shared_ptr<class Frame> pFrame;
 	 typedef yuri::shared_ptr<const class Frame> pcFrame;

 	 class Parameters;
 	 typedef yuri::shared_ptr<class Parameters> pParameters;

 	 class Parameter;
	 typedef yuri::shared_ptr<class Parameter> pParameter;

 	 class Pipe;
 	 typedef yuri::shared_ptr<class Pipe> pPipe;

 	 class IOThread;
 	 typedef yuri::shared_ptr<class IOThread> pIOThread;

 	 class RawVideoFrame;
	 typedef yuri::shared_ptr<class RawVideoFrame> pRawVideoFrame;

	 class CompressedVideoFrame;
	 typedef yuri::shared_ptr<class CompressedVideoFrame> pCompressedVideoFrame;
}


}


#endif /* FORWARD_H_ */
