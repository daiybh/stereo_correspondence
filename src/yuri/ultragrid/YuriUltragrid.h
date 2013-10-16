/*
 * YuriUltragrid.h
 *
 *  Created on: 16.10.2013
 *      Author: neneko
 */

#ifndef YURIULTRAGRID_H_
#define YURIULTRAGRID_H_

#include "yuri/core/utils/new_types.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/forward.h"
#include "yuri/log/Log.h"
#include "types.h"



namespace yuri {
namespace ultragrid {

codec_t yuri_to_uv(format_t);
format_t uv_to_yuri(codec_t );
std::string uv_to_string(codec_t);

core::pFrame copy_from_from_uv(const video_frame*, log::Log&);


}
}

namespace std {
template<>
struct hash<codec_t> {
 size_t operator()(const codec_t& x) const
{
	 return hash<int>()(x);
}
};

}


#endif /* YURIULTRAGRID_H_ */
