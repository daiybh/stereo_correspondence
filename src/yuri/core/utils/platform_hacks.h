#ifdef YURI_STD_TO_STRING_MISSING
#include <cstdio>
#include <string>
namespace std {
inline std::string to_string( int value )
{
	char buf[100];
	std::sprintf(buf, "%d", value);
	return buf;
}
inline std::string to_string( long value )
{
	char buf[100];
	std::sprintf(buf, "%ld", value);
	return buf;
}
inline std::string to_string( long long value )
{
	char buf[100];
	std::sprintf(buf, "%lld", value);
	return buf;
}
inline std::string to_string( unsigned value )
{
	char buf[100];
	std::sprintf(buf, "%u", value);
	return buf;
}
inline std::string to_string( unsigned long value )
{
	char buf[100];
	std::sprintf(buf, "%lu", value);
	return buf;
}
inline std::string to_string( unsigned long long value )
{
	char buf[100];
	std::sprintf(buf, "%llu", value);
	return buf;
}
inline std::string to_string( float value )
{
	char buf[100];
	std::sprintf(buf, "%f", value);
	return buf;
}
inline std::string to_string( double value )
{
	char buf[100];
	std::sprintf(buf, "%f", value);
	return buf;
}
inline std::string to_string( long double value ) 
{
	char buf[100];
	std::sprintf(buf, "%Lf", value);
	return buf;
}
}
#endif

#ifdef YURI_STD_STRTOI_MISSING
#include <cstdlib>
namespace std {
inline int strtoi(const char* str, char** str_end, int base)
{
	return static_cast<int>(strtol(str, str_end, base));
}
}
#endif


#if defined(YURI_STD_STOI_MISSING) || defined(YURI_STD_STOL_MISSING) || defined(YURI_STD_STOUL_MISSING)
#include <cstdlib>
#include <string>
namespace std {

#ifdef YURI_STD_STOL_MISSING
inline long stol(const std::string& s)
{
	return std::strtol(s.c_str(), nullptr, 10);
}
#endif

#ifdef YURI_STD_STOI_MISSING
inline int stoi(const std::string& s)
{
	return std::strtoi(s.c_str(), nullptr, 10);
}
#endif


#ifdef YURI_STD_STOUL_MISSING
inline unsigned long stoul(const std::string& s) 
{ 
	return std::strtoul(s.c_str(), nullptr, 10); 
}
#endif

}
#endif

