/*
 * register.cpp
 *
 *  Created on: 28.10.2013
 *      Author: neneko
 */

#include "DeckLinkInput.h"
#include "DeckLinkOutput.h"
#include "yuri/core/thread/IOThreadGenerator.h"
#include "yuri/core/thread/InputRegister.h"

MODULE_REGISTRATION_BEGIN("decklink")
	REGISTER_IOTHREAD("decklink_input",yuri::decklink::DeckLinkInput)
	REGISTER_INPUT_THREAD("decklink_input", yuri::decklink::DeckLinkBase::enumerate_inputs)
	REGISTER_IOTHREAD("decklink_output",yuri::decklink::DeckLinkOutput)
MODULE_REGISTRATION_END()




