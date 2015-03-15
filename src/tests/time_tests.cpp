/*
 * time_tests.cpp
 *
 *  Created on: 15. 3. 2015
 *      Author: neneko
 */


#include "catch.hpp"
#include "yuri/core/utils/time_types.h"


TEST_CASE( "Duration_t constructors", "[duration_t]" ) {
	using namespace yuri;
    REQUIRE( duration_t(1) == 1_us );
    REQUIRE( 1_ms == 1000_us );
    REQUIRE( 1000_ms == 1_s );
    REQUIRE( 60_s == 1_minutes );
    REQUIRE( 60_minutes == 1_hours );
    REQUIRE( 24_hours == 1_days );
}

TEST_CASE( "Duration_t operators", "[duration_t]" ) {
	using namespace yuri;
    REQUIRE( ( 1_us + 1_us) == 2_us );
    REQUIRE( ( 23_hours + 59_minutes + 59_s + 999_ms + 1000_us ) == 1_days);
    REQUIRE( ( 3_ms - 1000_us) == 2_ms );
}

