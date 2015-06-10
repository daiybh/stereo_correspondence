/*!
 * @file 		test_string_generator.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		20. 5. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */


#include "catch.hpp"
#include "yuri/core/utils/string_generator.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/frame/raw_frame_types.h"

namespace yuri{
namespace core{

TEST_CASE("string generator", "[sg]" ) {
	SECTION("simple") {
		REQUIRE(utils::generate_string("") == "");
		REQUIRE(utils::generate_string("ahoj") == "ahoj");

	}
	if (utils::is_extended_generator_supported()) {
		SECTION("sequence") {
			REQUIRE(utils::generate_string("%s",543) == "543");
			REQUIRE(utils::generate_string("%01s",543) == "543");
			REQUIRE(utils::generate_string("%5s",543) == "  543");
			REQUIRE(utils::generate_string("%06s",543) == "000543");
		}
		auto f = RawVideoFrame::create_empty(raw_format::rgb24, resolution_t{800,600});
		f->set_index(1);
		SECTION("index") {
			REQUIRE(utils::generate_string("%i",0,f) == "1");
			REQUIRE(utils::generate_string("%01i",543,f) == "1");
			REQUIRE(utils::generate_string("%5i",543,f) == "    1");
			REQUIRE(utils::generate_string("%06i",543,f) == "000001");
		}
		SECTION("frame parameters") {
			REQUIRE(utils::generate_string("%r",0,f) == "800x600");
			REQUIRE(utils::generate_string("%f",0,f) == "RGB");
			REQUIRE(utils::generate_string("%F",0,f) == "RGB 24 bit");
		}
	}
}


}
}



