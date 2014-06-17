#include <iostream>

// Optix
#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>
#include <ImageLoader.h>

// Physx
#include <PxPhysicsAPI.h>
#include <extensions/PxExtensionsAPI.h> 
#include <extensions/PxDefaultErrorCallback.h>
#include <extensions/PxDefaultAllocator.h> 
#include <extensions/PxDefaultSimulationFilterShader.h>
#include <extensions/PxDefaultCpuDispatcher.h>
#include <extensions/PxShapeExt.h>
#include <foundation/PxMat33.h> 
#include <extensions/PxSimpleFactory.h>

// Optix Sample Helpers
#include <sutil.h>
#include <GLUTDisplay.h>

#include "commonStructs.h"
#include "Helpers.h"

using namespace optix;

// Optix Vars
Helpers helpers;

std::string project_name = "BilliardsBlur";

// Physx Vars
static physx::PxPhysics* gPhysicsSDK = NULL;
static physx::PxDefaultErrorCallback gDefaultErrorCallback;
static physx::PxDefaultAllocator gDefaultAllocatorCallback;
static physx::PxSimulationFilterShader gDefaultFilterShader=physx::PxDefaultSimulationFilterShader;

physx::PxScene* gScene = NULL;
physx::PxReal myTimestep = 1.0f/60.0f;
const float gravity = -9.8f;

void createActors();
void convertMat(physx::PxMat33 m, physx::PxVec3 t, float* mat);

// Common Vars
float ball_radius = 1.0f;
float total_balls = 5; // Go to createBalls to add more to the vector
float3 balls_pos[5] = { 
	make_float3(0.0f,0.0f,0.0f),
	make_float3(ball_radius * 2, 0.0f, 0.0f),
	make_float3(-ball_radius * 2, 0.0f, 0.0f),
	make_float3(0.0f, 0.0f, ball_radius * 2),
	make_float3(-10.0f, 0.0f, -10.0f)
};

std::vector<physx::PxRigidActor*> balls_actors;
optix::Group balls_group;

//------------------------------------------------------------------------------
//
// Pollball Helpers
//
//------------------------------------------------------------------------------

inline float deg_to_rad( float degrees )
{
  return degrees * M_PIf / 180.0f;
}

inline float rad_to_deg( float radians )
{
  return radians * 180.0f / M_PIf;
}

optix::Matrix3x3 Rotation(float3 rotation)
{
	float alpha = -deg_to_rad( rotation.x );
	float beta  = -deg_to_rad( rotation.y );
	float gamma = -deg_to_rad( rotation.z );

	float s_a = sinf(alpha);
	float c_a = cosf(alpha);

	float s_b = sinf(beta);
	float c_b = cosf(beta);

	float s_g = sinf(gamma);
	float c_g = cosf(gamma);

	float rotate_x[3*3] = {   1,    0,    0,
							0,   c_a, -s_a,
							0,   s_a,  c_a };

	float rotate_y[3*3] = {  c_b,   0,   s_b,
							0,    1,    0,
							-s_b,   0,   c_b };

	float rotate_z[3*3] = {  c_g, -s_g,   0,
							s_g,  c_g,   0, 
							0,    0,    1 };

	optix::Matrix3x3 mat_x(rotate_x);
	optix::Matrix3x3 mat_y(rotate_y);
	optix::Matrix3x3 mat_z(rotate_z);

	optix::Matrix3x3 mat = mat_z * mat_y * mat_x;

	return mat;
}

class PoolBall
{
public:

  PoolBall() {};
  PoolBall( Context context, const std::string& texture    = "",
            float              radius     = 1.0f,
			float              position_x = 0.0f,
            float              position_y = 0.0f,
            float              position_z = 0.0f,
            float              rotation_x = 0.0f,
            float              rotation_y = 0.0f,
            float              rotation_z = 0.0f,
            float              color_r    = 1.0f,
            float              color_g    = 1.0f,
            float              color_b    = 1.0f );

  Material& getMaterial() { return material; }
  Geometry& getGeometry() { return geometry; }

private:

  Context  m_context;
  Material material;
  Geometry geometry;

public:

  float  radius;
  float3 rotation;
  float3 position;
  float3 color;

  void createGeometry();
  void createMaterial();
  void createTexture(const std::string& filename);
};

PoolBall::PoolBall(Context context, const std::string& texture, float radius,
				   float position_x, float position_y, float position_z,
				   float rotation_x, float rotation_y, float rotation_z,
				   float color_r, float color_g, float color_b)
				   : m_context(context),
				   radius(radius),
				   position(make_float3(position_x, position_y, position_z)),
				   rotation(make_float3(rotation_x, rotation_y, rotation_z)),
				   color(make_float3(color_r, color_g, color_b))
{
	createGeometry();
	createMaterial();
	createTexture(texture);
}

void PoolBall::createGeometry()
{
	geometry = m_context->createGeometry();

	geometry->setPrimitiveCount( 1 );
	std::string ptx_path = helpers.getPTXPath(project_name, "sphere_texcoord.cu");
	geometry->setBoundingBoxProgram(  m_context->createProgramFromPTXFile( ptx_path, "bounds" ) );
	geometry->setIntersectionProgram( m_context->createProgramFromPTXFile( ptx_path, "intersect" ) );
	geometry["sphere"]->setFloat( 0.0f, 0.0f, 0.0f, radius );

	optix::Matrix3x3 matrix = Rotation(rotation);

	geometry["matrix_row_0"]->setFloat(matrix[0*3+0], matrix[0*3+1], matrix[0*3+2]);
	geometry["matrix_row_1"]->setFloat(matrix[1*3+0], matrix[1*3+1], matrix[1*3+2]);
	geometry["matrix_row_2"]->setFloat(matrix[2*3+0], matrix[2*3+1], matrix[2*3+2]);
}

void PoolBall::createMaterial()
{
	std::string ptx_path = helpers.getPTXPath(project_name, "clearcoat.cu" );
	Program clearcoat_ch = m_context->createProgramFromPTXFile( ptx_path, "closest_hit_radiance" );
	Program clearcoat_ah = m_context->createProgramFromPTXFile( ptx_path, "any_hit_shadow" );

	material = m_context->createMaterial();

	material->setClosestHitProgram( 0, clearcoat_ch );
	material->setAnyHitProgram( 1, clearcoat_ah );

	material["importance_cutoff"]->setFloat( 1e-2f );
	material["cutoff_color"]->setFloat( 0.34f, 0.55f, 0.85f );
	material["fresnel_exponent"]->setFloat( 4.0f );
	material["fresnel_minimum"]->setFloat( 0.1f );
	material["fresnel_maximum"]->setFloat( 1.0f );
	material["reflection_color"]->setFloat( 1.0f, 1.0f, 1.0f );
	material["reflection_maxdepth"]->setInt( 5 );

	material["Ka"]->setFloat(0.2f, 0.2f, 0.2f);
	material["Ks"]->setFloat(1.0f, 1.0f, 1.0f);
	material["exponent"]->setFloat(128.0f);

	material["Kd"]->setFloat(color.x, color.y, color.z);
}

void PoolBall::createTexture(const std::string& filename)
{
  const float3 default_color = make_float3(1.0f, 1.0f, 1.0f);
  material["kd_map"]->setTextureSampler( loadTexture( m_context, filename, default_color) );
}

//------------------------------------------------------------------------------
//
// BilliardsBlur definition
//
//------------------------------------------------------------------------------

class BilliardsBlurScene : public SampleScene
{
public:

	BilliardsBlurScene(const std::string& texture_path)
		: m_texture_path(texture_path){};

	virtual void initScene( InitialCameraData& camera_data );
	virtual void trace( const RayGenCameraData& camera_data );
	virtual optix::Buffer getOutputBuffer();
	virtual bool keyPressed(unsigned char key, int x, int y);

	static bool m_useGLBuffer;
	static bool m_animate;
	static bool m_motionBlur;

	static float aperture;
	static float distance_offset;

private:
	void createGeometry();
	void createBalls();
	void updateGeometry();
	void StepPhysX();
	std::string texpath(const std::string& base);

	std::string m_texture_path;
	std::vector<PoolBall> poolballs;

	const static unsigned WIDTH;
	const static unsigned HEIGHT;
};

bool BilliardsBlurScene::m_useGLBuffer = true;
bool BilliardsBlurScene::m_animate = true;
bool BilliardsBlurScene::m_motionBlur = true;

float BilliardsBlurScene::aperture = 0.0f;
float BilliardsBlurScene::distance_offset = 0.0f;

const unsigned BilliardsBlurScene::WIDTH  = 1024u;
const unsigned BilliardsBlurScene::HEIGHT = 728u;

/////////////////////////////////////////////////////////////////////////////////////////////////
// InitScene

void BilliardsBlurScene::initScene( InitialCameraData& camera_data )
{
	// Setup state
	m_context->setRayTypeCount( 2 );
	m_context->setEntryPointCount( 1 );
	m_context->setStackSize( 1440 );

	// Limit number of devices to 1 as this is faster for this particular sample.
	std::vector<int> enabled_devices = m_context->getEnabledDevices();
	m_context->setDevices(enabled_devices.begin(), enabled_devices.begin()+1);

	// Setup output buffer
	m_context["output_format"]->setInt( RT_FORMAT_FLOAT4 );

	m_context["output_buffer_f4"]->setBuffer( createOutputBuffer(RT_FORMAT_FLOAT4, WIDTH, HEIGHT));
    m_context["output_buffer_f3"]->setBuffer( m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, 1, 1));

	// Setup camera
	camera_data = InitialCameraData( make_float3( 10.3f, 10.0f, -10.0f ), // eye
									 make_float3( 0.5f, 0.3f,  1.0f ),   // lookat
									 make_float3( 0.0f, 1.0f,  0.0f ),  // up
									              40.0f );             // vfov

	// Declare camera variables.  The values do not matter, they will be overwritten in trace.
	m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
	m_context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
	m_context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
	m_context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );

	// Context variables
	m_context["radiance_ray_type"]->setUint(0);
	m_context["shadow_ray_type"]->setUint(1);
	m_context["bad_color"]->setFloat( 0.0f, 1.0f, 0.0f );
	m_context["bg_color"]->setFloat( make_float3( 0.34f, 0.55f, 0.85f ) );
	m_context["scene_epsilon"]->setFloat( 1.e-2f );
	m_context["max_depth"]->setInt(5);

	// Camera parameters
	m_context["focal_scale"]->setFloat( 0.0f ); // Value is set in trace()
	m_context["aperture_radius"]->setFloat(aperture);

	// Setup programs
	std::string ptx_path = helpers.getPTXPath(project_name, "dof_camera.cu" );
	Program ray_gen_program = m_context->createProgramFromPTXFile( ptx_path, "dof_camera" );
	m_context->setRayGenerationProgram( 0, ray_gen_program );
	Program exception_program = m_context->createProgramFromPTXFile( ptx_path, "exception" );
	m_context->setExceptionProgram( 0, exception_program );
	m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( helpers.getPTXPath(project_name, "constantbg.cu" ), "miss" ) );

	// Setup lighting
	BasicLight lights[] = {
		{ make_float3( -30.0f, -10.0f, 80.0f ), make_float3( 1.0f, 1.0f, 1.0f ), 1 },
		{ make_float3(  10.0f,  30.0f, 20.0f ), make_float3( 1.0f, 1.0f, 1.0f ), 1 }
	};

	Buffer light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
	light_buffer->setFormat(RT_FORMAT_USER);
	light_buffer->setElementSize(sizeof(BasicLight));
	light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
	memcpy(light_buffer->map(), lights, sizeof(lights));
	light_buffer->unmap();

	m_context["lights"]->set(light_buffer);
	m_context["ambient_light_color"]->setFloat( 0.4f, 0.4f, 0.4f );

	// Rendering variables
	m_context["frame_number"]->setUint(1);
	m_context["jitter"]->setFloat(0.0f, 0.0f, 0.0f, 0.0f);

	// Setup secene
	createGeometry();
	m_context->validate();
	m_context->compile();
}

void BilliardsBlurScene::createGeometry()
{
	//////////////////////////////////////////////////////////////
	// Floor
	//////////////////////////////////////////////////////////////

	Geometry parallelogram = m_context->createGeometry();
	parallelogram->setPrimitiveCount( 1u );
	std::string ptx_path = helpers.getPTXPath(project_name, "parallelogram.cu");
	parallelogram->setBoundingBoxProgram( m_context->createProgramFromPTXFile( ptx_path, "bounds" ) );
	parallelogram->setIntersectionProgram( m_context->createProgramFromPTXFile( ptx_path, "intersect" ) );

	float3 anchor = optix::make_float3( -20.0f, 0.01f, 20.0f);
	float3 v1 = optix::make_float3( 40, 0, 0);
	float3 v2 = optix::make_float3( 0, 0, -40);

	float3 normal = cross( v1, v2 );
	normal = normalize( normal );
	float d = dot( normal, anchor );
	v1 *= 1.0f/dot( v1, v1 );
	v2 *= 1.0f/dot( v2, v2 );
	float4 plane = make_float4( normal, d );
  
	parallelogram["plane"]->setFloat( plane );
	parallelogram["v1"]->setFloat( v1 );
	parallelogram["v2"]->setFloat( v2 );
	parallelogram["anchor"]->setFloat( anchor );

	// Material
	Program check_ch = m_context->createProgramFromPTXFile( helpers.getPTXPath(project_name, "distributed_phong.cu" ), "closest_hit_radiance" );
	Program check_ah = m_context->createProgramFromPTXFile( helpers.getPTXPath(project_name, "distributed_phong.cu" ), "any_hit_shadow" );

	Material floor_matl = m_context->createMaterial();
	floor_matl->setClosestHitProgram( 0, check_ch );
	floor_matl->setAnyHitProgram( 1, check_ah );

	floor_matl["phong_exp"]->setFloat( 0.0f );
	floor_matl["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f);

	// Texture scaling
	floor_matl["Ka"]->setFloat( 0.35f, 0.35f, 0.35f );
	floor_matl["Kd"]->setFloat( 0.50f, 0.50f, 0.50f );
	floor_matl["Ks"]->setFloat( 1.00f, 1.00f, 1.00f );

	// Floor texture
	floor_matl["ka_map"]->setTextureSampler(loadTexture(m_context, texpath("cloth.ppm"), make_float3(1, 1, 1)));
	floor_matl["kd_map"]->setTextureSampler(loadTexture(m_context, texpath("cloth.ppm"), make_float3(1, 1, 1)));
	floor_matl["ks_map"]->setTextureSampler(loadTexture(m_context, "", make_float3(0, 0, 0)));

	//////////////////////////////////////////////////////////////
	// Balls
	//////////////////////////////////////////////////////////////

	createBalls();

	// Geometry group
	std::vector<Transform> ball_ts;
	for (unsigned i=0; i < total_balls; i++)
	{
		// GeometryInstance
		GeometryInstance gi = m_context->createGeometryInstance(poolballs[i].getGeometry(), &(poolballs[i].getMaterial()),&(poolballs[i].getMaterial())+1 );

		// GeometryGroup
		GeometryGroup box_group = m_context->createGeometryGroup();
		box_group->setChildCount( 1u );
		box_group->setChild( 0, gi );
		box_group->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );

		// Transform
		Transform box_transform = m_context->createTransform();
		box_transform->setChild(box_group);

		ball_ts.push_back(box_transform);
	}

	balls_group = m_context->createGroup();
	balls_group->setChildCount( static_cast<unsigned int>(ball_ts.size()) );
	for ( unsigned int i = 0; i < ball_ts.size(); ++i ) { 
		balls_group->setChild( i, ball_ts[i] );
	}
	balls_group->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );

	GeometryInstance floor_gi = m_context->createGeometryInstance( parallelogram, &floor_matl, &floor_matl+1 );
	GeometryGroup floor_group = m_context->createGeometryGroup();
	floor_group->setChildCount( 1u );
	floor_group->setChild( 0, floor_gi );
	floor_group->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );

	// Main Group
	Group main_group = m_context->createGroup();
	main_group->setChildCount(2u);
	main_group->setChild(0, balls_group);
	main_group->setChild(1, floor_group);
	main_group->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );

	m_context["top_object"]->set( main_group );
	m_context["top_shadower"]->set( main_group );
}

void BilliardsBlurScene::createBalls()
{
	poolballs.push_back( PoolBall(m_context, texpath("pool_1.ppm"),  1.00f,  -00.872f-90.0f,  28.322f, -14.422f,  1.00f, 0.96f, 0.94f) );
	poolballs.push_back( PoolBall(m_context, texpath("pool_9.ppm"),  1.00f,   11.339f-90.0f,  14.126f, -17.257f,  1.00f, 0.96f, 0.94f) );
	poolballs.push_back( PoolBall(m_context, texpath("pool_8.ppm"),  1.00f,   15.492f-90.0f,  16.372f, -21.111f,  1.00f, 0.96f, 0.94f) );
	poolballs.push_back( PoolBall(m_context, texpath("pool_4.ppm"),  1.00f,   27.679f-90.0f, -02.611f, -16.905f,  1.00f, 0.96f, 0.94f) );
	poolballs.push_back( PoolBall(m_context, ""           , 1.00f,   00.000f,        00.000f,  00.000f,  1.00f, 0.96f, 0.58f) );
}

std::string BilliardsBlurScene::texpath(const std::string& base)
{
  return m_texture_path + "/" + base;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
// Trace

void BilliardsBlurScene::trace( const RayGenCameraData& camera_data )
{
	m_context["eye"]->setFloat( camera_data.eye );
	m_context["U"]->setFloat( camera_data.U );
	m_context["V"]->setFloat( camera_data.V );
	m_context["W"]->setFloat( camera_data.W );

	float focal_distance = length(camera_data.W) + distance_offset;
	focal_distance = fmaxf(focal_distance, m_context["scene_epsilon"]->getFloat());
	float focal_scale = focal_distance / length(camera_data.W);
	m_context["focal_scale"]->setFloat( focal_scale );

	Buffer buffer = getOutputBuffer();
	RTsize buffer_width, buffer_height;
	buffer->getSize( buffer_width, buffer_height );

	if(m_animate){
		// Update PhysX	
		if (gScene) 
			StepPhysX(); 

		// Update Geometry
		updateGeometry();
	}

	if(m_motionBlur) {
		static unsigned frame = 1;

		if( m_camera_changed ) {
			frame = 1;
			m_camera_changed = false;
		}

		if(frame > 10)
			frame = 10;

		m_context["frame_number"]->setUint(frame++);
	} else {
		m_context["frame_number"]->setUint(1);
	}

	m_context->launch( 0,
		static_cast<unsigned int>(buffer_width),
		static_cast<unsigned int>(buffer_height)
		);
}

void BilliardsBlurScene::StepPhysX() 
{ 
	gScene->simulate(myTimestep);        
	       
	while(!gScene->fetchResults()){}
} 

void BilliardsBlurScene::updateGeometry()
{
	for( unsigned int i = 0; i < total_balls; ++i ) {
		optix::Transform transform = balls_group->getChild<optix::Transform>(i);
		physx::PxRigidActor* ball_actor = balls_actors.at(i);

		physx::PxU32 nShapes = ball_actor->getNbShapes(); 
		physx::PxShape** shapes=new physx::PxShape*[nShapes];
	
		ball_actor->getShapes(shapes, nShapes);     
		while (nShapes--) 
		{ 
			physx::PxShape* shape = shapes[nShapes];
			physx::PxTransform pT = physx::PxShapeExt::getGlobalPose(*shape, *ball_actor);
			physx::PxMat33 m = physx::PxMat33(pT.q );
			float mat[16];
			convertMat(m,pT.p, mat);
			transform->setMatrix(false, mat, NULL);
		} 
		delete [] shapes;
	}

	balls_group->getAcceleration()->markDirty();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
// getOutputBuffer

Buffer BilliardsBlurScene::getOutputBuffer()
{
  return m_context["output_buffer_f4"]->getBuffer();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
// keyPressed

bool BilliardsBlurScene::keyPressed(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'a': {
		if( m_animate )
		{
			std::cout << "Stoping animation..." << std::endl;
			m_animate = false;
		}
		else
		{
			std::cout << "Starting animation..." << std::endl;
			m_animate = true;
		}
		return true;
			  }
	case 'b': {
		if( m_motionBlur )
		{
			std::cout << "Motion Blur disabled..." << std::endl;
			m_motionBlur = false;
		}
		else
		{
			std::cout << "Motion Blur enable..." << std::endl;
			m_motionBlur = true;
		}
		return true;
			  }
	}
   return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////


//------------------------------------------------------------------------------
//
//  Physx
//
//------------------------------------------------------------------------------

void initializePhysx()
{
	// Init Physx
	physx::PxFoundation* foundation = PxCreateFoundation( PX_PHYSICS_VERSION, gDefaultAllocatorCallback, gDefaultErrorCallback );
	if( !foundation )
		std::cerr << "PxCreateFoundation failed!" << std::endl;

	gPhysicsSDK = PxCreatePhysics( PX_PHYSICS_VERSION, *foundation, physx::PxTolerancesScale() );
	if( gPhysicsSDK == NULL )
	{
		std::cerr << "Error creating PhysX3 device." << std::endl;
		std::cerr << "Exiting.." << std::endl;
		exit(1);
	}

	if( !PxInitExtensions( *gPhysicsSDK ))
		std::cerr << "PxInitExtensions failed!" << std::endl;

	// Create Scene
	physx::PxSceneDesc sceneDesc( gPhysicsSDK->getTolerancesScale() );
	sceneDesc.gravity = physx::PxVec3( 0.0f, gravity, 0.0f );

	if( !sceneDesc.cpuDispatcher )
	{
        physx::PxDefaultCpuDispatcher* mCpuDispatcher = physx::PxDefaultCpuDispatcherCreate(1);

        if( !mCpuDispatcher )
           std::cerr << "PxDefaultCpuDispatcherCreate failed!" << std::endl;

        sceneDesc.cpuDispatcher = mCpuDispatcher;
    } 
 	if( !sceneDesc.filterShader )
		sceneDesc.filterShader  = gDefaultFilterShader;

	gScene = gPhysicsSDK->createScene(sceneDesc);
	if( !gScene )
        std::cerr << "createScene failed!" << std::endl;

	gScene->setVisualizationParameter(physx::PxVisualizationParameter::eSCALE, 1.0);
	gScene->setVisualizationParameter(physx::PxVisualizationParameter::eCOLLISION_SHAPES, 1.0f);

	// Create Actors
	createActors();
}

void createActors()
{
	// Create Material
	physx::PxMaterial* ballMaterial = gPhysicsSDK->createMaterial(0.6f, 0.1f, 0.6f);
	physx::PxMaterial* planeMaterial = gPhysicsSDK->createMaterial(0.5f, 0.5f, 0.5f);

	// Create Floor
	physx::PxReal d = 0.0f;
	physx::PxTransform pose = physx::PxTransform( physx::PxVec3( 0.0f, 0, 0.0f ), physx::PxQuat( physx::PxHalfPi, physx::PxVec3( 0.0f, 0.0f, 1.0f )));

	physx::PxRigidStatic* plane = gPhysicsSDK->createRigidStatic(pose);
	if (!plane)
			std::cerr << "create plane failed!" << std::endl;

	physx::PxShape* shape = plane->createShape(physx::PxPlaneGeometry(), *planeMaterial);
	if (!shape)
		std::cerr << "create shape failed!" << std::endl;
	gScene->addActor(*plane);
	

	// Create Balls
	physx::PxReal density = 2.0f;
	physx::PxTransform ballTransform(physx::PxVec3(0.0f, 0.0f, 0.0f));
	physx::PxSphereGeometry ballGeometry(ball_radius);
		
	for(unsigned int i = 0; i < total_balls; ++i)
	{
		ballTransform.p = physx::PxVec3(balls_pos[i].x, balls_pos[i].y, balls_pos[i].z);

		physx::PxRigidDynamic *ballActor = PxCreateDynamic(*gPhysicsSDK, ballTransform, ballGeometry, *ballMaterial, density);
		ballActor->setAngularDamping(0.2f);
		ballActor->setLinearDamping(0.01f);
		ballActor->setMass(1.0f);
		if(i == (total_balls - 1))
			ballActor->setLinearVelocity(physx::PxVec3(60.0f, 0.0f, 60.0f)); 

		gScene->addActor(*ballActor);

		balls_actors.push_back(ballActor);
	}
}

void convertMat(physx::PxMat33 m, physx::PxVec3 t, float* mat)
{
   mat[0] = m.column0[0];
   mat[1] = m.column1[0];
   mat[2] = m.column2[0];
   mat[3] = t[0];

   mat[4] = m.column0[1];
   mat[5] = m.column1[1];
   mat[6] = m.column2[1];
   mat[7] = t[1];

   mat[8] = m.column0[2];
   mat[9] = m.column1[2];
   mat[10] = m.column2[2];
   mat[11] = t[2];

   mat[12] = 0;
   mat[13] = 0;
   mat[14] = 0;
   mat[15] = 1;
}

//------------------------------------------------------------------------------
//
//  Main
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"

    << "  -h  | --help                               Print this usage message\n"
    << "  -P  | --pbo                                Use OpenGL PBO for output buffer (default)\n"
    << "  -n  | --nopbo                              Use internal output buffer\n"
	<< "  --noanimate                                Disables Animation\n"
	<< "  --nomotionblur                             Disables MotionBlur\n"
    << std::endl;
  GLUTDisplay::printUsage();

  std::cerr
    << "App keystrokes:\n"
	<< "  a Toggles animation\n"
	<< std::endl;

  if ( doExit ) exit(1);
}

int main( int argc, char** argv )
{
	// Init Physx
	initializePhysx();

	GLUTDisplay::init( argc, argv );

	for(int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if(arg == "-P" || arg == "--pbo") {
			BilliardsBlurScene::m_useGLBuffer = true;
		} else if( arg == "-n" || arg == "--nopbo" ) {
			BilliardsBlurScene::m_useGLBuffer = false;
		} else if( arg == "--noanimate" ) {
			BilliardsBlurScene::m_animate = false;
		} else if( arg == "--nomotionblur" ) {
			BilliardsBlurScene::m_animate = false;
		} else if( arg == "-h" || arg == "--help" ) {
			printUsageAndExit(argv[0]);
		} else {
			std::cerr << "Unknown option '" << arg << "'\n";
			printUsageAndExit(argv[0]);
		}
	}

	if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

	try {
		BilliardsBlurScene scene("data/");
		GLUTDisplay::run( "BilliardsBlur", &scene, GLUTDisplay::CDAnimated );
	} catch( Exception& e ) {
	    sutilReportError( e.getErrorString().c_str() );    
	    exit(2);
	}

	return 0;
}