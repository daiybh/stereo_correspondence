/*!
 * @file 		forward.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		22.2.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
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

	 class Convert;
	 typedef yuri::shared_ptr<class Convert> pConvert;
}


}


#endif /* FORWARD_H_ */
