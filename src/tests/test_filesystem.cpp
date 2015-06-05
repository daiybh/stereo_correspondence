/*!
 * @file 		test_filesystem.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		20. 5. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#include "catch.hpp"
#include "yuri/core/utils/DirectoryBrowser.h"


namespace yuri{
namespace core{

TEST_CASE("filesystem", "[fs]" ) {
	SECTION("path manipulation") {
		REQUIRE(filesystem::get_directory("/tmp/abc/ce.rere") == "/tmp/abc");
		REQUIRE(filesystem::get_directory("../ce.rere") == "..");
		REQUIRE(filesystem::get_directory("ce.rere") == "");
		REQUIRE(filesystem::get_filename("/tmp/abc/ce.rere") == "ce.rere");
		REQUIRE(filesystem::get_filename("/tmp/abc/ce.rere", false) == "ce");

	}
}


}
}


