/*
 * compressed_frame_params.h
 *
 *  Created on: 4.10.2013
 *      Author: neneko
 */

#ifndef COMPRESSED_FRAME_PARAMS_H_
#define COMPRESSED_FRAME_PARAMS_H_
#include "yuri/core/utils/new_types.h"
#include <string>
#include <vector>
#include <map>
namespace yuri {
namespace core {
namespace compressed_frame {


struct compressed_frame_info_t
{
	compressed_frame_info_t(format_t format, std::string name, std::vector<std::string> short_names, std::vector<std::string> mime_types, std::string fourcc = std::string()):
		format(format),name(name),short_names(short_names), mime_types(mime_types), fourcc(fourcc)
	{}
	format_t					format;
	std::string 				name;
	std::vector<std::string> 	short_names;
	std::vector<std::string> 	mime_types;
	std::string 				fourcc;
};

using comp_format_info_map_t = std::map<format_t, compressed_frame_info_t>;

struct formats {
	comp_format_info_map_t::const_iterator begin() const;
	comp_format_info_map_t::const_iterator end() const;
};

const compressed_frame_info_t &get_format_info(format_t format);
bool add_format(const compressed_frame_info_t &);
format_t new_user_format();
format_t parse_format(const std::string& name);
const std::string& get_format_name(format_t format);

}
}
}


#endif /* COMPRESSED_FRAME_PARAMS_H_ */
