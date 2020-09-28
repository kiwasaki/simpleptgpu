#include<iostream>
#include<vector>
#include<memory>

#include<curand.h>
#include<curand_kernel.h>

#include"ray.hpp"
#include"sphere.hpp"
#include"scene.hpp"
#include"image.hpp"

//
__global__ void make_scene( sphere **device_spheres, scene **device_scene, const int n )
{
	if( threadIdx.x == 0 && blockIdx.x == 0 ) {
		device_spheres[ 0 ] = new sphere( make_float3( 0.f, 1.0f, 10.f ), 1.f, make_float4( 1.f, 1.f, 1.f, 1.f ) );
		device_spheres[ 1 ] = new sphere( make_float3( 0.f, -1e3f, 0.f ), 1e3f, make_float4( 1.f, 1.f, 1.f, 1.f ) );
		device_spheres[ 2 ] = new sphere( make_float3( 1e3f + 5.f, 0.f, 0.f ), 1e3f, make_float4( 0.8f, 0.2f, 0.1f, 1.f ) ); //right
		device_spheres[ 3 ] = new sphere( make_float3( - 1e3f - 5.f, 0.f, 0.f ), 1e3f, make_float4( 1.f, 1.f, 1.f, 1.f ) ); //left
		device_spheres[ 4 ] = new sphere( make_float3( 0.f, 0.f, 1e3f + 15.f ), 1e3f, make_float4( 1.f, 1.f, 1.f, 1.f ) );
		device_spheres[ 5 ] = new sphere( make_float3( 0.f, 1e3f + 8, 0.f ), 1e3f, make_float4( 1.f, 1.f, 1.f, 1.f ) );
		device_spheres[ 6 ] = new sphere( make_float3( 0.f, 7.f, 10.f ), 1.f, make_float4( 10.f, 10.f, 10.f, - 1.f ) );
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
		d = { m_p * ( x - width / 2.f + curand_uniform( &rng ) ), m_p * ( y - height / 2.f + curand_uniform( &rng ) ), 1.f };
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
__device__ inline bool is_emissive( const intersection &isect )
{
	return ( isect.m_c.w < 0.f );
}

// sample direction for diffuse BRDF
__device__ inline float3 sample( const float3 n, const float xi1, const float xi2, float &pdf_w )
{
	const float3 t = normalize( ( std::abs( n.x ) > std::abs( n.y ) )? cross( n, make_float3( 0.f, n.z, - n.y ) ) : cross( n, make_float3( - n.z, 0.f, n.x ) ) );
	const float3 b = cross( t, n );
	const float cth = sqrt( 1.f - xi1 );
	const float sth = sqrt( max( 0.f, 1.f - cth * cth ) );
	const float cph = cos( 2.f * 3.14159265f * xi2 );
	const float sph = sin( 2.f * 3.14159265f * xi2 );
	pdf_w = 1.f / 3.14159265f * cth;
	return cth * n + sth * cph * t + sth * sph * b;
}

//
__global__ void render( float *pixels, scene **scene, curandState *rand_state, const int width, const int height, const float3 eye, const int ns )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if( ( x >= width ) && ( y >= height ) ) return;


	float3 L;
	float pdf_w;
	curandState rng = rand_state[ y * width + x ];
	const float m_p = 2.f * tan( 40.f / 2.f * 3.14159265f / 180.f ) / float( height );
	L = make_float3( 0.f, 0.f, 0.f );

	for( int i = 0; i < ns; ++i ) {
		intersection isect;
		ray r;
		float3 d;
		d = { m_p * ( x - width / 2.f + curand_uniform( &rng ) ), m_p * ( y - height / 2.f + curand_uniform( &rng ) ), 1.f };
		r = { eye, normalize( d ) };
		float3 tp = make_float3( 1.f, 1.f, 1.f );

		if( ( *scene )->intersect( r, isect ) ) {
			//
			if( isect.m_c.w < 0.f ) {
				L.x += isect.m_c.x;
				L.y += isect.m_c.y;
				L.z += isect.m_c.z;
			} else {
				for( int j = 0; j < 10; ++j ) {
					d = normalize( sample( isect.m_n, curand_uniform( &rng ), curand_uniform( &rng ), pdf_w ) );
					tp *= make_float3( isect.m_c.x / 3.14159265f, isect.m_c.y / 3.14159265f, isect.m_c.z / 3.14159265f ) * dot( d, isect.m_n ) / pdf_w;
					r = { isect.m_p, d };
					bool hit = ( *scene )->intersect( r, isect );
					if( hit ) {
						if( isect.m_c.w < 0.f ) {
							L += tp * make_float3( isect.m_c.x / 3.14159265f, isect.m_c.y / 3.14159265f, isect.m_c.z / 3.14159265f );
							break;
						}
					} else {
						break;
					}
				}
			}
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

	curand_init( 1984, y * width + x, 0, &rand_state[ y * width + x ] );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv )
{
	constexpr int width		= 512;
	constexpr int height	= 512;

	constexpr int n_object	= 7;

	float 						*device_image;
	std::unique_ptr< float [] > host_image;
	sphere **device_spheres;
	scene  **device_scene;
	curandState *device_rand_state;

	//
	const float3 eye = { 0.f, 3.f, - 5.f };

	//
	checkCudaErrors( cudaMalloc( ( void** ) &device_rand_state, sizeof( curandState ) * width * height ) );

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
		checkCudaErrors( cudaMalloc( ( void ** ) &device_rand_state, width * height * sizeof( curandState ) ) );
		init<<< grid, block >>>( device_rand_state, width, height );
		checkCudaErrors( cudaGetLastError() );
		checkCudaErrors( cudaDeviceSynchronize() );
	}


	{
		dim3 grid, block;
		block.x = 16;
		block.y = 16;
		grid.x = width  / block.x;
		grid.y = height / block.y;

		//trace<<< grid, block >>>( device_image, device_scene, width, height, eye );
		//render_aa<<< grid, block >>>( device_image, device_scene, device_rand_state, width, height, eye, 10 );
		render<<< grid, block >>>( device_image, device_scene, device_rand_state, width, height, eye, 4096 );
		checkCudaErrors( cudaGetLastError() );
		checkCudaErrors( cudaDeviceSynchronize() );
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
	checkCudaErrors( cudaFree( device_rand_state ) );
	checkCudaErrors( cudaFree( device_spheres ) );
	checkCudaErrors( cudaFree( device_scene ) );

	return 0;

}
