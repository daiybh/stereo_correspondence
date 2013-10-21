/*
 * raw_audio_frame_params.h
 *
 *  Created on: 21.10.2013
 *      Author: neneko
 */

#ifndef RAW_AUDIO_FRAME_PARAMS_H_
#define RAW_AUDIO_FRAME_PARAMS_H_
#include "yuri/core/utils/new_types.h"
#include <vector>
#include <map>

namespace yuri {
namespace core {
namespace raw_audio_format {
struct raw_audio_format_t;

const raw_audio_format_t &get_format_info(format_t format);
bool add_format(const raw_audio_format_t &);
format_t new_user_format();
format_t parse_format(const std::string& name);
const std::string& get_format_name(format_t format);
using format_info_map_t = std::map<format_t, raw_audio_format_t>;





struct raw_audio_format_t {
  raw_audio_format_t(format_t format, std::string long_name,
               std::vector<std::string> short_names, size_t bits_per_sample,
               bool little_endian = true)
      : format(format), name(long_name), short_names(short_names),
        bits_per_sample(bits_per_sample),little_endian(little_endian) {}
  ~raw_audio_format_t() noexcept {}
  format_t format;
  std::string name;
  std::vector<std::string> short_names;
  size_t	bits_per_sample;
  bool little_endian;

};

}
}
}


#endif /* RAW_AUDIO_FRAME_PARAMS_H_ */
