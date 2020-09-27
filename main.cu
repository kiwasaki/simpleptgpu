#include<iostream>
#include<vector>
#include<memory>

#include"ray.hpp"
#include"sphere.hpp"
#include"scene.hpp"
#include"image.hpp"

//
__global__ void make_scene( sphere **device_spheres, scene **device_scene, const int n )
{
	if( threadIdx.x == 0 && blockIdx.x == 0 ) {
		device_spheres[ 0 ] = new sphere( make_float3( 0.f, 1.0f, 10.f ), 1.f, make_float4( 1.f, 1.f, 1.f, 1.f ) );
		device_spheres[ 1 ] = new sphere( make_float3( 0.f, -1e4f, 0.f ), 1e4f, make_float4( 1.f, 1.f, 0.f, 1.f ) );
		device_spheres[ 2 ] = new sphere( make_float3( 0.f, 7.f, 10.f ), 1.f, make_float4( 10.f, 0.f, 0.f, 0.f ) );
		*device_scene = new scene( device_spheres, n );
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



int main( int argc, char** argv )
{
	constexpr int width		= 512;
	constexpr int height	= 512;

	constexpr int n_object	= 3;

	float 						*device_image;
	std::unique_ptr< float [] > host_image;
	sphere **device_spheres;
	scene  **device_scene;

	//curandState *device_rand_state;

	//
	const float3 eye = { 0.f, 3.f, - 5.f };

	//
	//checkCudaErrors( cudaMalloc( ( void** ) &device_rand_state, sizeof( curandState ) * width * height ) );

	//
	host_image = std::make_unique< float [] >( 3 * width * height );
	checkCudaErrors( cudaMalloc( ( void ** ) &device_image, sizeof( float ) * 3 * width * height ) );

	//シーンの作成
	{
		checkCudaErrors( cudaMalloc( ( void ** ) &device_spheres, n_object * sizeof( sphere * ) ) );
		checkCudaErrors( cudaMalloc( ( void ** ) &device_scene  , 1 * sizeof( scene *  ) ) );
		make_scene<<< 1, 1 >>>( device_spheres, device_scene, n_object );
		checkCudaErrors( cudaGetLastError() );
		checkCudaErrors( cudaDeviceSynchronize() );
	}

	{
		dim3 grid, block;
		block.x = 16;
		block.y = 16;
		grid.x = width  / block.x;
		grid.y = height / block.y;

		trace<<< grid, block >>>( device_image, device_scene, width, height, eye );
	}

	{
		free_scene<<< 1, 1 >>>( device_spheres, device_scene, n_object );
		checkCudaErrors( cudaGetLastError() );
		checkCudaErrors( cudaDeviceSynchronize() );
	}

	checkCudaErrors( cudaMemcpy( host_image.get(), device_image, sizeof( float ) * 3 * width * height, cudaMemcpyDeviceToHost ) );

	save_bmp( host_image.get(), width, height, "test.bmp" );

	//delete
	checkCudaErrors( cudaFree( device_image ) );
	checkCudaErrors( cudaFree( device_spheres ) );
	checkCudaErrors( cudaFree( device_scene ) );

	return 0;

}
