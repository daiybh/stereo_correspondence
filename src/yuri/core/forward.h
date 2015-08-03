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
 	 using pThreadChild 	= std::shared_ptr<ThreadChild> ;

 	 class ThreadBase;
 	 using pwThreadBase 	= std::weak_ptr<class ThreadBase>;
 	 using pThreadBase 		= std::shared_ptr<class ThreadBase>;
 	 using pcThreadBase		= std::shared_ptr<class ThreadBase const>;

 	 class Frame;
 	 using pFrame 			= std::shared_ptr<class Frame>;
 	 using pcFrame			= std::shared_ptr<const class Frame>;

 	 class Parameters;
 	 using pParameters		= std::shared_ptr<class Parameters>;

 	 class Parameter;
	 using pParameter 		= std::shared_ptr<class Parameter>;

 	 class Pipe;
 	 using pPipe			= std::shared_ptr<class Pipe>;

 	 class IOThread;
 	 using pIOThread		= std::shared_ptr<class IOThread>;

 	 class RawVideoFrame;
	 using pRawVideoFrame	= std::shared_ptr<class RawVideoFrame>;

	 class CompressedVideoFrame;
	 using pCompressedVideoFrame
			 	 	 	 	= std::shared_ptr<class CompressedVideoFrame>;

	 class Convert;
	 using pConvert 		= std::shared_ptr<class Convert>;
}


}


#endif /* FORWARD_H_ */
