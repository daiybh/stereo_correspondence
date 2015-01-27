/*!
 * @file 		compressed_frame_params.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		4.10.2013
 * @date		21.11.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
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
	EXPORT compressed_frame_info_t(format_t format, std::string name, std::vector<std::string> short_names, std::vector<std::string> mime_types, std::string fourcc = std::string()):
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
	EXPORT comp_format_info_map_t::const_iterator begin() const;
	EXPORT comp_format_info_map_t::const_iterator end() const;
};

EXPORT const compressed_frame_info_t &get_format_info(format_t format);
EXPORT bool add_format(const compressed_frame_info_t &);
EXPORT format_t new_user_format();
EXPORT format_t parse_format(const std::string& name);
EXPORT const std::string& get_format_name(format_t format);
EXPORT format_t get_format_from_mime(const std::string& mime);



inline const std::string& get_format_name(format_t format) { return get_format_info(format).name; }
}
}
}


#endif /* COMPRESSED_FRAME_PARAMS_H_ */
