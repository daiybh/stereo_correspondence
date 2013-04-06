/*
 * DeckLink3DVideoFrame.cpp
 *
 *  Created on: Jan 25, 2012
 *      Author: worker
 */

#include "DeckLink3DVideoFrame.h"

namespace yuri {

namespace decklink {

bool operator==(const REFIID & first, const REFIID & second)
{
	const yuri::ubyte_t*r1=reinterpret_cast<const yuri::ubyte_t*>(&first);
	const yuri::ubyte_t*r2=reinterpret_cast<const yuri::ubyte_t*>(&second);
	for (yuri::uint_t i = 0; i < 16; ++i) if (*r1++!=*r2++) return false;
	return true;
}

DeckLink3DVideoFrame::DeckLink3DVideoFrame(yuri::usize_t width, yuri::usize_t height, BMDPixelFormat format, BMDFrameFlags flags)
:width(width),height(height),format(format),buffer(0),flags(flags),
 packing(bmdVideo3DPackingFramePacking)
{

	if (format == bmdFormat8BitYUV) linesize_ = width*2;
	else if (format == bmdFormat8BitARGB || format==bmdFormat8BitBGRA) linesize_ = width*4;
	else if (format == bmdFormat10BitYUV) {
		linesize_ = (width/6 + (width%6?1:0))*16;
	}
	buffer = new yuri::ubyte_t[height*linesize_];

}

DeckLink3DVideoFrame::~DeckLink3DVideoFrame()
{

}

long DeckLink3DVideoFrame::GetWidth ()
{
	return static_cast<long>(width);
}
long DeckLink3DVideoFrame::GetHeight ()
{
	return static_cast<long>(height);
}
long DeckLink3DVideoFrame::GetRowBytes ()
{
	//std::cerr << "GetRowBytes " << __FILE__ << ":" << __LINE__ << std::endl;
	return linesize_;
}
BMDPixelFormat DeckLink3DVideoFrame::GetPixelFormat ()
{
	//std::cerr << "get format " << __FILE__ << ":" << __LINE__ << std::endl;
	return format;
}
BMDFrameFlags DeckLink3DVideoFrame::GetFlags ()
{
	return flags;
}
HRESULT DeckLink3DVideoFrame::GetBytes (void **buffer)
{
	//std::cerr << "GetBytes " << __FILE__ << ":" << __LINE__ << std::endl;
	*buffer = this->buffer;
	return S_OK;
}

HRESULT STDMETHODCALLTYPE DeckLink3DVideoFrame::QueryInterface(REFIID iid, void **ppv)
{
	//int id = iid.byte0;
	//std::cerr << "QueryInterface " << hex << id<< dec << ": " << __FILE__ << ":" << __LINE__ << std::endl;
	if (iid == IID_IDeckLinkVideoFrame3DExtensions)	{
		*ppv = dynamic_cast<IDeckLinkVideoFrame3DExtensions*>(this);
		return S_OK;
	}
	return E_NOINTERFACE;
}
HRESULT DeckLink3DVideoFrame::GetTimecode (/* in */ BMDTimecodeFormat format, /* out */ IDeckLinkTimecode **timecode)
{
	//std::cerr << "GetTimecode " << __FILE__ << ":" << __LINE__ << std::endl;
	return S_OK;
}
HRESULT DeckLink3DVideoFrame::GetAncillaryData (/* out */ IDeckLinkVideoFrameAncillary **ancillary)
{
	//std::cerr << "GetAncillaryData " << __FILE__ << ":" << __LINE__ << std::endl;
	return S_OK;
}
BMDVideo3DPackingFormat DeckLink3DVideoFrame::Get3DPackingFormat ()
{
	//std::cerr << "Get3DPackingFormat " << __FILE__ << ":" << __LINE__ << std::endl;
	return packing;
}
HRESULT DeckLink3DVideoFrame::GetFrameForRightEye (/* out */ IDeckLinkVideoFrame* *rightEyeFrame)
{
	//std::cerr << "GetFrameForRightEye " << __FILE__ << ":" << __LINE__ << std::endl;
	*rightEyeFrame = right.get();
	return S_OK;
}
void DeckLink3DVideoFrame::add_right(shared_ptr<DeckLink3DVideoFrame> r)
{
	right = r;
}
void DeckLink3DVideoFrame::set_packing_format(BMDVideo3DPackingFormat fmt)
{
	packing = fmt;
}
shared_ptr<DeckLink3DVideoFrame> DeckLink3DVideoFrame::get_right()
{
	return right;
}

}

}
