/*!
 * @file 		raw_frame_params.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		15.9.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef RAW_FRAME_PARAMS_H_
#define RAW_FRAME_PARAMS_H_
#include "yuri/core/utils/new_types.h"
#include <vector>
#include <string>
#include <map>

namespace yuri {
namespace core {
namespace raw_format {
struct raw_format_t;

EXPORT const raw_format_t &get_format_info(format_t format);
EXPORT bool add_format(const raw_format_t &);
EXPORT format_t new_user_format();
EXPORT format_t parse_format(const std::string& name);
EXPORT const std::string& get_format_name(format_t format);
using format_info_map_t = std::map<format_t, raw_format_t>;

struct EXPORT formats {
	format_info_map_t::const_iterator begin() const;
	format_info_map_t::const_iterator end() const;
};
/*
 * Plane components: (case sensitive)
 * ? - Unused
 * R - Red
 * G - Green
 * B - Blue
 * A - Alpha
 * Y - Luminance
 * U - Cr color difference
 * V - Cb color difference
 * x - CIE 1931 X Component
 * y - CIE 1931 Y Component
 * z - CIE 1931 Z Component
 */

struct plane_info_t {
  plane_info_t(std::string components, std::pair<size_t, size_t> bit_depth,
               std::vector<size_t> component_bit_depths, size_t sub_x = 1,
               size_t sub_y = 1, size_t alignment_requirement = 0)
      : components(components), bit_depth(bit_depth), component_bit_depths(component_bit_depths), sub_x(sub_x),
        sub_y(sub_y), alignment_requirement(alignment_requirement) {
  }
  ~plane_info_t() noexcept {}
  /// Minimal repeating subset of components. Can be empty if there's no expressible pattern.
  std::string components;

  /// Bit depth per repeating subset components. Second component is number of pixels it represents
  std::pair<size_t, size_t> bit_depth; // Depth per pixel (as Num/Den)
  /// Bit depths per each component in @em components. Sum should be less or equal to @em bit_depth.first
  std::vector<size_t> component_bit_depths;
  size_t sub_x;
  size_t sub_y;
  size_t alignment_requirement;
};

struct raw_format_t {
  raw_format_t(format_t format, std::string long_name,
               std::vector<std::string> short_names, std::string fourcc,
               std::vector<plane_info_t> planes)
      : format(format), name(long_name), short_names(short_names),
        fourcc(fourcc), planes(planes) {}
  ~raw_format_t() noexcept {}
  format_t format;
  std::string name;
  std::vector<std::string> short_names;
  std::string fourcc;
  std::vector<plane_info_t> planes;

};


inline const std::string& get_format_name(format_t format) { return get_format_info(format).name; }

inline size_t get_fmt_bpp(format_t fmt, size_t plane) {
	const auto& fi = get_format_info(fmt);
	const auto& bpp = fi.planes[plane].bit_depth;
	return bpp.first / bpp.second;
}

}
}
}

#endif /* RAW_FRAME_PARAMS_H_ */
