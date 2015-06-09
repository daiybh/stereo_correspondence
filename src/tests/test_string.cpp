/*!
 * @file 		test_string.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		7. 6. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */


#include "catch.hpp"
#include "yuri/core/utils/string.h"

namespace yuri{
namespace core{


TEST_CASE("string", "" ) {
	SECTION("split") {
		std::vector<std::string> empty_vector;
		REQUIRE(utils::split_string("", ';') == empty_vector);
		std::vector<std::string> zero_3_vector = {"", "", ""};
		REQUIRE(utils::split_string(";;;", ';') == zero_3_vector);
		const std::string a1="hello;world";
		std::vector<std::string> a2 = {"hello", "world"};
		REQUIRE(utils::split_string(a1, ';') == a2);
		std::vector<std::string> trlll{"tr","l","l","l"};
		REQUIRE(utils::split_string("tralalala", 'a') == trlll);
	}
}

}
}
