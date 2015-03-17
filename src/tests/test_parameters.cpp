/*!
 * @file 		test_parameters.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		17.03.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "catch.hpp"
#include "yuri/core/parameter/Parameters.h"


namespace yuri{
namespace core{

namespace {
const std::string desc = "sample description";
const std::string desc2 = "new sample description xxx";
const std::string desc3 = "another description";
const std::string param_name = "sample name";

const auto test_value1 = int{37};
const auto test_value2 = float{0.125f};
const auto test_value3 = double{0.1};
const auto test_value4 = std::string{"sample text"};
const auto test_value5 = true;
const auto test_value6 = false;

const auto test_value7 = resolution_t{800,600};

template<class T>
struct remove_all {
	using type = typename std::remove_reference<typename std::remove_cv<T>::type>::type;
};

}


TEST_CASE( "parameter", "[parameter]" ) {
	SECTION("name") {
		Parameter p(param_name);
		REQUIRE( p.get_name() == param_name );

		Parameter p2("");
		REQUIRE( p2.get_name() == "" );
	}
	SECTION("description") {
		Parameter p(param_name, 0, desc);
		REQUIRE( p.get_description() == desc );
		p.set_description(desc2);
		REQUIRE( p.get_description() == desc2 );
		p[desc3];
		REQUIRE( p.get_description() == desc3 );
	}
	SECTION("single values") {
#define TEST_VALUE(value) \
		{\
			Parameter p0(param_name, value); \
			REQUIRE( p0.get<remove_all<decltype(value)>::type>() == value); \
			Parameter p1(param_name); \
			p1 = value; \
			REQUIRE( p1.get<remove_all<decltype(value)>::type>() == value); \
		}

		// Simple values

		TEST_VALUE(test_value1)
		TEST_VALUE(test_value2)
		TEST_VALUE(test_value3)
		TEST_VALUE(test_value4)
		TEST_VALUE(test_value5)
		TEST_VALUE(test_value6)

		// Compound values
		TEST_VALUE(test_value7)
#undef TEST_VALUE
	}


	SECTION("multiple_values") {
		// This feature is currently basically unused and may need some rework

#define TEST_VALUE(p, index, val) \
		REQUIRE( p.get_indexed<remove_all<decltype(val)>::type>(index) == val );


		std::vector<event::pBasicEvent> v0 = {std::make_shared<event::EventInt>(test_value1),
											std::make_shared<event::EventDouble>(test_value2),
											std::make_shared<event::EventDouble>(test_value3),
											std::make_shared<event::EventString>(test_value4),
											std::make_shared<event::EventBool>(test_value5),
											std::make_shared<event::EventBool>(test_value6)};

		Parameter p0(param_name);
		p0.set_value(std::make_shared<event::EventVector>(v0));

		TEST_VALUE(p0, 0, test_value1)
		TEST_VALUE(p0, 1, test_value2)
		TEST_VALUE(p0, 2, test_value3)
		TEST_VALUE(p0, 3, test_value4)
		TEST_VALUE(p0, 4, test_value5)
		TEST_VALUE(p0, 5, test_value6)

		std::map<std::string, event::pBasicEvent> m0 = {
											{"v0", std::make_shared<event::EventInt>(test_value1)},
											{"v1", std::make_shared<event::EventDouble>(test_value2)},
											{"v2", std::make_shared<event::EventDouble>(test_value3)},
											{"v3", std::make_shared<event::EventString>(test_value4)},
											{"v4", std::make_shared<event::EventBool>(test_value5)},
											{"v5", std::make_shared<event::EventBool>(test_value6)}};

		Parameter p1(param_name);
		p1.set_value(std::make_shared<event::EventDict>(m0));

		TEST_VALUE(p1, "v0", test_value1)
		TEST_VALUE(p1, "v1", test_value2)
		TEST_VALUE(p1, "v2", test_value3)
		TEST_VALUE(p1, "v3", test_value4)
		TEST_VALUE(p1, "v4", test_value5)
		TEST_VALUE(p1, "v5", test_value6)
#undef TEST_VALUE

	}
}


TEST_CASE( "parameters", "[parameters]" ) {
	Parameters params;
	REQUIRE( params.get_description() == "" );
	params.set_description(desc);
	REQUIRE( params.get_description() == desc );
	params.set_description(desc2);
	REQUIRE( params.get_description() == desc2 );

#define TEST_VALUE(name, desc, value) \
		params[name][desc]=value; \
		REQUIRE( params[name].get<remove_all<decltype(value)>::type>() == value ); \
		REQUIRE( params[name].get_description() == desc );

	TEST_VALUE("v1", desc, test_value1)
	TEST_VALUE("v2", desc, test_value2)
	TEST_VALUE("v3", desc, test_value3)
	TEST_VALUE("v4", desc, test_value4)
	TEST_VALUE("v5", desc, test_value5)
	TEST_VALUE("v6", desc, test_value6)
	TEST_VALUE("v7", desc, test_value7)

	// There should be 7 values in params
	REQUIRE( std::distance(params.begin(), params.end()) == 7 );

#undef TEST_VALUE

}

}
}

