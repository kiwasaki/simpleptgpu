#include<string>
#include<fstream>


inline void save_bmp( const float *pixels, const int width, const int height, const std::string &filename )
{
	std::ofstream ofs( filename, std::ios::binary );

	const int row_bytes = ((width * 3 + 3) >> 2) << 2;

	const short bfType = 0x4d42;
	ofs.write((char*)&bfType, 2);
	const int bfSize = 14 + 40 + row_bytes * height;
	ofs.write((char*)&bfSize, 4);
	const short bfReserved1 = 0;
	ofs.write((char*)&bfReserved1, 2);
	const short bfReserved2 = 0;
	ofs.write((char*)&bfReserved2, 2);
	const int bfOffBits = 14 + 40;
	ofs.write((char*)&bfOffBits, 4);

	const int biSize = 40;
	ofs.write((char*)&biSize, 4);
	const int biWidth = width;
	ofs.write((char*)&biWidth, 4);
	const int biHeight = height;
	ofs.write((char*)&biHeight, 4);
	const short biPlanes = 1;
	ofs.write((char*)&biPlanes, 2);
	const short biBitCount = 24;
	ofs.write((char*)&biBitCount, 2);
	const int biCompression = 0;
	ofs.write((char*)&biCompression, 4);
	const int biSizeImage = 0;
	ofs.write((char*)&biSizeImage, 4);
	const int biXPelsPerMeter = 0;
	ofs.write((char*)&biXPelsPerMeter, 4);
	const int biYPelsPerMeter = 0;
	ofs.write((char*)&biYPelsPerMeter, 4);
	const int biClrUsed = 0;
	ofs.write((char*)&biClrUsed, 4);
	const int biClrImportant = 0;
	ofs.write((char*)&biClrImportant, 4);

	std::vector<unsigned char> scanline(
			row_bytes
	);
	for(int y = 0; y < height; y++){
		for(int x = 0; x < width; x++){
			scanline[3 * x + 0] = std::max( 0, std::min( 255, int( 255 * pow( pixels[ 3 * ( y * width + x ) + 2 ], 1.f / 2.2f ) ) ) );
			scanline[3 * x + 1] = std::max( 0, std::min( 255, int( 255 * pow( pixels[ 3 * ( y * width + x ) + 1 ], 1.f / 2.2f ) ) ) );
			scanline[3 * x + 2] = std::max( 0, std::min( 255, int( 255 * pow( pixels[ 3 * ( y * width + x ) + 0 ], 1.f / 2.2f ) ) ) );
		}
		ofs.write((char*)scanline.data(), row_bytes);
	}
}