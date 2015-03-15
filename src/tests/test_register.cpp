/*
 * test_register.cpp
 *
 *  Created on: 15. 3. 2015
 *      Author: neneko
 */

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "yuri/log/Log.h"

#include "yuri/core/thread/IOThreadGenerator.h"

yuri::core::Parameters conf()
{
	yuri::core::Parameters p;
	p["test"]=123456;
	return p;
}


std::shared_ptr<yuri::core::IOThread> gen(yuri::log::Log & /* log */, yuri::core::pwThreadBase /* parent */, const yuri::core::Parameters& /* params */)
{
	return {};
}

TEST_CASE( "IOThread register", "[register]" ) {
	using namespace yuri;
	auto g = IOThreadGenerator::get_instance();
	auto count0 = g.list_keys().size();
	SECTION("Startup") {
		REQUIRE ( count0 > 0 );
	}
	SECTION("Registration") {
		const auto name = std::string{"test_xxxxxx"};
		REQUIRE( !g.is_registered(name) );
		g.register_generator(name, gen, conf);
		REQUIRE( g.is_registered(name) );
		auto count1 = g.list_keys().size();
		REQUIRE( count1 == ( count0 + 1 ) );
		auto cfg = g.configure(name);
		REQUIRE( cfg["test"].get<int>() == 123456 );
	}
}
