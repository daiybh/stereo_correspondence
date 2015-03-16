/*!
 * @file 		test_make_list.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		16. 3. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */


#include "catch.hpp"
#include "yuri/core/utils/make_list.h"


namespace yuri{
namespace core{
TEST_CASE( "make_list", "[make_list]" ) {
	std::vector<int> vi = {1,2,3,4,5};
	auto si = utils::make_list(vi, ", ");
	REQUIRE( si == std::string{"1, 2, 3, 4, 5"});
	auto si2 = utils::make_list(vi, "");
	REQUIRE( si2 == std::string{"12345"});
	auto si3 = utils::make_list(std::vector<int>{0}, "xxxxx");
	REQUIRE( si3 == std::string{"0"});

	std::vector<std::string> vs = {"hello", "cruel", "world"};
	auto ss = utils::make_list(vs, " ");
	REQUIRE( ss == std::string{"hello cruel world"});
}


TEST_CASE( "print_list", "[print_list]" ) {
	std::vector<int> vi = {1,2,3,4,5};
	std::stringstream ss1;
	utils::print_list(ss1, vi, ", ");
	REQUIRE( ss1.str() == std::string{"1, 2, 3, 4, 5"});
	std::stringstream ss2;
	utils::print_list(ss2, vi, "");
	REQUIRE( ss2.str() == std::string{"12345"});
	std::stringstream ss3;
	utils::print_list(ss3, std::vector<int>{0}, "xxxxx");
	REQUIRE( ss3.str() == std::string{"0"});

	std::vector<std::string> vs = {"hello", "cruel", "world"};
	std::stringstream ss4;
	utils::print_list(ss4, vs, " ");
	REQUIRE( ss4.str() == std::string{"hello cruel world"});
}

}
}


