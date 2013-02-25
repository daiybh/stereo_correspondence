/*
 * DeckLink3DVideoFrame.h
 *
 *  Created on: Jan 25, 2012
 *      Author: worker
 */

#ifndef DECKLINK3DVIDEOFRAME_H_
#define DECKLINK3DVIDEOFRAME_H_
#include <DeckLinkAPI.h>
#include "yuri/core/BasicFrame.h"
namespace yuri {

namespace decklink {
bool operator==(const REFIID & first, const REFIID & second);

class DeckLink3DVideoFrame: public IDeckLinkVideoFrame, public IDeckLinkVideoFrame3DExtensions{
public:
	DeckLink3DVideoFrame(yuri::usize_t width, yuri::usize_t height, BMDPixelFormat format, BMDFrameFlags flags);
	virtual ~DeckLink3DVideoFrame();
	virtual ULONG STDMETHODCALLTYPE		AddRef ()									{return 1;}
	virtual ULONG STDMETHODCALLTYPE		Release ()									{return 1;}
	virtual long GetWidth (void);
	virtual long GetHeight (void);
	virtual long GetRowBytes (void);
	virtual BMDPixelFormat GetPixelFormat (void);
	virtual BMDFrameFlags GetFlags (void);
	virtual HRESULT GetBytes (/* out */ void **buffer);
	virtual HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppv);
	virtual HRESULT GetTimecode (/* in */ BMDTimecodeFormat format, /* out */ IDeckLinkTimecode **timecode);
	virtual HRESULT GetAncillaryData (/* out */ IDeckLinkVideoFrameAncillary **ancillary);
    virtual BMDVideo3DPackingFormat Get3DPackingFormat (void);
    virtual HRESULT GetFrameForRightEye (/* out */ IDeckLinkVideoFrame* *rightEyeFrame);

    void add_right(shared_ptr<DeckLink3DVideoFrame> r);
    void set_packing_format(BMDVideo3DPackingFormat fmt);
    shared_ptr<DeckLink3DVideoFrame> get_right();
protected:
    yuri::usize_t width, height;
    BMDPixelFormat format;
    yuri::ubyte_t *buffer;
    yuri::size_t bpp;
    BMDFrameFlags flags;
    shared_ptr<DeckLink3DVideoFrame> right;
    BMDVideo3DPackingFormat packing;
};

}

}

#endif /* DECKLINK3DVIDEOFRAME_H_ */
