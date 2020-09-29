#include<iostream>
#include<vector>
#include<memory>
#include<chrono>

#include<curand.h>
#include<curand_kernel.h>

#include"ray.hpp"
#include"sphere.hpp"
#include"scene.hpp"
#include"image.hpp"

constexpr float pi = 3.14159265f;
constexpr float two_pi = 2.f * pi;
constexpr float inv_pi = 0.318309886f;

//
__global__ void make_scene( sphere **device_spheres, scene **device_scene, const int n )
{
	if( threadIdx.x == 0 && blockIdx.x == 0 ) {
		int k = 0;

		//device_spheres[ 0 ] = new sphere( make_float3( 0.f, 1.0f, 10.f ), 1.f, make_float4( 1.f, 1.f, 1.f, 0.1f ) );
		device_spheres[ k++ ] = new sphere( make_float3( 0.f, -1e3f, 0.f ), 1e3f, make_float4( 1.f, 1.f, 1.f, 0.1f ) ); //floor
		device_spheres[ k++ ] = new sphere( make_float3( 1e3f + 5.f, 0.f, 0.f ), 1e3f, make_float4( 0.8f, 0.2f, 0.1f, 0.1f ) ); //right
		device_spheres[ k++ ] = new sphere( make_float3( - 1e3f - 5.f, 0.f, 0.f ), 1e3f, make_float4( 0.1f, 0.2f, 0.8f, 0.1f ) ); //left
		device_spheres[ k++ ] = new sphere( make_float3( 0.f, 0.f, 1e3f + 15.f ), 1e3f, make_float4( 1.f, 1.f, 1.f, 0.1f ) ); //far
		device_spheres[ k++ ] = new sphere( make_float3( 0.f, 1e3f + 8, 0.f ), 1e3f, make_float4( 1.f, 1.f, 1.f, 0.01f ) ); //ceil
		device_spheres[ k++ ] = new sphere( make_float3( 0.f, 7.0f, 10.f ), 1.f, make_float4( 10.f, 10.f, 10.f, - 1.f ) ); //light source
		device_spheres[ k++ ] = new sphere( make_float3( 0.f, 1.0f, 10.f ), 1.f, make_float4( 1.f, 1.f, 1.f, 0.9f ) );

		*device_scene = new scene( device_spheres, k );
	}
}


//
__global__ void free_scene( sphere **device_spheres, scene **device_scene, const int n )
{
	if( threadIdx.x == 0 && blockIdx.x == 0 ) {
		for( int i = 0; i < n; ++i ) {
			delete device_spheres[ i ];
		}
		delete *device_scene;
	}
}

//
__global__ void trace( float *pixels, scene **scene, const int width, const int height, const float3 eye )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if( ( x >= width ) && ( y >= height ) ) return;

	const float m_p = 2.f * tan( 40.f / 2.f * 3.14159265f / 180.f ) / float( height );
	const float3 d = { m_p * ( x - width / 2.f ), m_p * ( y - height / 2.f ), 1.f };
	const ray r = { eye, normalize( d ) };

	intersection isect;

	bool hit = ( *scene )->intersect( r, isect );
	if( hit ) {
		pixels[ 3 * ( y * width + x ) + 0 ] = 0.5f * ( isect.m_n.x + 1.f );
		pixels[ 3 * ( y * width + x ) + 1 ] = 0.5f * ( isect.m_n.y + 1.f );
		pixels[ 3 * ( y * width + x ) + 2 ] = 0.5f * ( isect.m_c.z + 1.f );
	} else {
		pixels[ 3 * ( y * width + x ) + 0 ] = 0.f;
		pixels[ 3 * ( y * width + x ) + 1 ] = 0.f;
		pixels[ 3 * ( y * width + x ) + 2 ] = 0.f;
	}
}

//
__global__ void render_aa( float *pixels, scene **scene, curandState *rand_state, const int width, const int height, const float3 eye, const int ns )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if( ( x >= width ) && ( y >= height ) ) return;

	intersection isect;
	float3 d, L;
	ray r;
	curandState rng = rand_state[ y * width + x ];

	const float m_p = 2.f * tan( 40.f / 2.f * 3.14159265f / 180.f ) / float( height );

	L = make_float3( 0.f, 0.f, 0.f );
	for( int i = 0; i < ns; ++i ) {
		d = { m_p * ( float( x ) - float( width ) / 2.f + curand_uniform( &rng ) ), m_p * ( float( y ) - float( height ) / 2.f + curand_uniform( &rng ) ), 1.f };
		//d = { m_p * ( x - width / 2.f + 0.5f ), m_p * ( y - height / 2.f + 0.5f ), 1.f };
		r = { eye, normalize( d ) };
		if( ( *scene )->intersect( r, isect ) ) {
			L.x += 0.5f * ( isect.m_n.x + 1.f );
			L.y += 0.5f * ( isect.m_n.y + 1.f );
			L.z += 0.5f * ( isect.m_n.z + 1.f );
		}
	}
	pixels[ 3 * ( y * width + x ) + 0 ] = L.x / float( ns );
	pixels[ 3 * ( y * width + x ) + 1 ] = L.y / float( ns );
	pixels[ 3 * ( y * width + x ) + 2 ] = L.z / float( ns );
}


//
__global__ void init( curandState *rand_state, const int width, const int height )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if( ( x >= width ) || ( y >= height ) ) return;

	//curand_init( 1984, y * width + x, 0, &rand_state[ y * width + x ] );
	curand_init( 1984 + y * width + x, 0, 0, &rand_state[ y * width + x ] );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv )
{
	constexpr int width  = 1024;
	constexpr int height = 1024;

	constexpr int n_object	= 7;

	float *device_buffer;

	sphere **device_spheres;
	scene  **device_scene;
	curandState *device_rand_state;

	//
	const float3 eye = { 0.f, 3.f, - 5.f };

	//
	checkCudaErrors( cudaMallocManaged( ( void ** ) &device_buffer, sizeof( float ) * 3 * width * height ) );

	//
	checkCudaErrors( cudaMalloc( ( void** ) &device_rand_state, sizeof( curandState ) * width * height ) );

	//シーンの作成
	{
		checkCudaErrors( cudaMalloc( ( void ** ) &device_spheres, n_object * sizeof( sphere * ) ) );
		checkCudaErrors( cudaMalloc( ( void ** ) &device_scene  , 1 * sizeof( scene *  ) ) );
		make_scene<<< 1, 1 >>>( device_spheres, device_scene, n_object );
		checkCudaErrors( cudaGetLastError() );
		checkCudaErrors( cudaDeviceSynchronize() );
	}

	const auto start = std::chrono::system_clock::now();

	{
		dim3 grid, block;
		block.x = 16;
		block.y = 16;
		grid.x = width  / block.x;
		grid.y = height / block.y;
		checkCudaErrors( cudaMalloc( ( void ** ) &device_rand_state, width * height * sizeof( curandState ) ) );
		init<<< grid, block >>>( device_rand_state, width, height );
		checkCudaErrors( cudaGetLastError() );
		checkCudaErrors( cudaDeviceSynchronize() );
	}

	{
		dim3 grid, block;
		block.x = 32;
		block.y = 32;
		grid.x = width  / block.x;
		grid.y = height / block.y;

		render_aa<<< grid, block >>>( device_buffer, device_scene, device_rand_state, width, height, eye, 10 );
		checkCudaErrors( cudaGetLastError() );
		checkCudaErrors( cudaDeviceSynchronize() );
	}
	const auto end = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast< std::chrono::milliseconds >( end - start ).count() << "ms.\n";

	{
		free_scene<<< 1, 1 >>>( device_spheres, device_scene, n_object );
		checkCudaErrors( cudaGetLastError() );
		checkCudaErrors( cudaDeviceSynchronize() );
	}

	//
	save_bmp( device_buffer, width, height, "rtaa.bmp" );

	checkCudaErrors( cudaFree( device_rand_state ) );
	checkCudaErrors( cudaFree( device_spheres ) );
	checkCudaErrors( cudaFree( device_scene ) );
	checkCudaErrors( cudaFree( device_buffer ) );

	return 0;

}
