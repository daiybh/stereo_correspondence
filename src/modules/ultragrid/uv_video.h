/*!
 * @file 		uv_video.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		13.2.2014
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */


#ifndef UV_VIDEO_H_
#define UV_VIDEO_H_
#include <memory>
#include "video.h"

// Using shared_ptr because it can erase the type of deleter
using video_frame_t = std::shared_ptr<video_frame>;

// Damned min/max macros AARGGGH
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#endif /* UV_VIDEO_H_ */
