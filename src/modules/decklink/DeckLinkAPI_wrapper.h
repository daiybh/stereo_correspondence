/*!
 * @file 		DeckLinkAPI_wrapper.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		12. 6. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef SRC_MODULES_DECKLINK_DECKLINKAPI_WRAPPER_H_
#define SRC_MODULES_DECKLINK_DECKLINKAPI_WRAPPER_H_


#if defined(__clang__) || (defined(__GNUC__) && (__GNUC__ > 4 || __GNUC_MINOR__ > 7))
#define CAN_DISABLE_PEDANTIC
#endif

#ifdef CAN_DISABLE_PEDANTIC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include "DeckLinkAPI.h"
#ifdef CAN_DISABLE_PEDANTIC
#pragma GCC diagnostic pop
#undef CAN_DISABLE_PEDANTIC
#endif



#endif /* SRC_MODULES_DECKLINK_DECKLINKAPI_WRAPPER_H_ */
