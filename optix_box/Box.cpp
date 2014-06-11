#include <iostream>

// Optix
#include <optix.h>
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

// Optix Vars
Helpers helpers;

std::string project_name = "optix_box";

// Physx Vars
static physx::PxPhysics* gPhysicsSDK = NULL;
static physx::PxDefaultErrorCallback gDefaultErrorCallback;
static physx::PxDefaultAllocator gDefaultAllocatorCallback;
static physx::PxSimulationFilterShader gDefaultFilterShader=physx::PxDefaultSimulationFilterShader;

physx::PxScene* gScene = NULL;
physx::PxRigidActor *box;
physx::PxReal myTimestep = 1.0f/60.0f;
const float gravity = -9.8;

void createActors();
void convertMat(physx::PxMat33 m, physx::PxVec3 t, float* mat);

// Common Vars
physx::PxVec3 box_size(1,1,1);


//------------------------------------------------------------------------------
//
// SimpleBoxPhysx definition
//
//------------------------------------------------------------------------------

class SimpleBoxPhysx : public SampleScene
{
public:
	virtual void initScene( InitialCameraData& camera_data );
	virtual void trace( const RayGenCameraData& camera_data );
	virtual optix::Buffer getOutputBuffer();
	virtual bool keyPressed(unsigned char key, int x, int y);

	void createGeometry();
	void updateGeometry();
	void StepPhysX();

	static bool m_useGLBuffer;
	static bool m_animate;
private:
	optix::Group m_main_group;

	const static int WIDTH;
	const static int HEIGHT;
};

bool SimpleBoxPhysx::m_useGLBuffer = true;
bool SimpleBoxPhysx::m_animate = true;
const int SimpleBoxPhysx::WIDTH  = 1024;
const int SimpleBoxPhysx::HEIGHT = 1024;

/////////////////////////////////////////////////////////////////////////////////////////////////
// InitScene

void SimpleBoxPhysx::initScene( InitialCameraData& camera_data )
{
	// Two Rays, on light and one shadow
	m_context->setRayTypeCount( 2 );
	m_context->setEntryPointCount( 1 );
	m_context->setStackSize( 1520 );

	m_context["max_depth"]->setInt( 6 );
    m_context["radiance_ray_type"]->setUint( 0u );
    m_context["shadow_ray_type"]->setUint( 1u );
    m_context["scene_epsilon"]->setFloat( 1.e-3f );

	// Output buffer
	optix::Variable output_buffer = m_context["output_buffer"];

    output_buffer->set(createOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4, WIDTH, HEIGHT ) );

	// Set up camera
    camera_data = InitialCameraData( optix::make_float3( 8.3f, 4.0f, -20.0f ), // eye
                                     optix::make_float3( 0.5f, 0.3f,  1.0f ), // lookat
                                     optix::make_float3( 0.0f, 1.0f,  0.0f ), // up
                                     60.0f );                          // vfov

    // Declare camera variables.  The values do not matter, they will be overwritten in trace.
    m_context["eye"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["U"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["V"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["W"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );

	// Ray generation program
	std::string ptx_path = helpers.getPTXPath( project_name, "pinhole_camera.cu" );
	optix::Program ray_gen_program = m_context->createProgramFromPTXFile( ptx_path, "pinhole_camera" );
	m_context->setRayGenerationProgram( 0, ray_gen_program );

	// Exception program
	optix::Program exception_program = m_context->createProgramFromPTXFile( ptx_path, "exception" );
    m_context->setExceptionProgram( 0, exception_program );
    m_context["bad_color"]->setFloat( 0.0f, 1.0f, 0.0f );

	// Miss program
	m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( helpers.getPTXPath( project_name, "constantbg.cu" ), "miss" ) );
    m_context["bg_color"]->setFloat( optix::make_float3(108.0f/255.0f, 166.0f/255.0f, 205.0f/255.0f) * 0.5f );

	// Setup lights
    m_context["ambient_light_color"]->setFloat(0,0,0);
    BasicLight lights[] = { 
      { { -7.0f, 15.0f, -7.0f }, { .8f, .8f, .8f }, 1 }
    };

    optix::Buffer light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
    light_buffer->setFormat(RT_FORMAT_USER);
    light_buffer->setElementSize(sizeof(BasicLight));
    light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
    memcpy(light_buffer->map(), lights, sizeof(lights));
    light_buffer->unmap();

    m_context["lights"]->set(light_buffer);

	// Create scene geometry
	createGeometry();

	// FInalize
	m_context->validate();
	m_context->compile();
}

void SimpleBoxPhysx::createGeometry()
{
	// Material programs
	const std::string transparency_ptx = helpers.getPTXPath( project_name, "transparent.cu" );
	optix::Program transparent_ch = m_context->createProgramFromPTXFile( transparency_ptx, "closest_hit_radiance" );
	optix::Program transparent_ah = m_context->createProgramFromPTXFile( transparency_ptx, "any_hit_shadow" );

	// Box programs
	const std::string box_ptx = helpers.getPTXPath( project_name, "box.cu" );
	optix::Program box_bounds    = m_context->createProgramFromPTXFile( box_ptx, "box_bounds" );
	optix::Program box_intersect = m_context->createProgramFromPTXFile( box_ptx, "box_intersect" );

	//////////////////////////////////////////////////////////////
	// Box
	//////////////////////////////////////////////////////////////

	// Geometry
	optix::Geometry box = m_context->createGeometry();
	box->setPrimitiveCount( 1u );
    box->setBoundingBoxProgram( box_bounds );
    box->setIntersectionProgram( box_intersect );

	box["boxmin"]->setFloat( -box_size.x, -box_size.y, -box_size.z );
    box["boxmax"]->setFloat( box_size.x, box_size.y, box_size.z );

	// Material
    optix::Material box_matl = m_context->createMaterial();
    box_matl->setClosestHitProgram( 0, transparent_ch );
    box_matl->setAnyHitProgram( 1, transparent_ah );

    box_matl["Kd"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );
    box_matl["Ks"]->setFloat( optix::make_float3( 0.2f, 0.2f, 0.2f ) );
    box_matl["Ka"]->setFloat( optix::make_float3( 0.05f, 0.05f, 0.05f ) );
    box_matl["phong_exp"]->setFloat(32);
    box_matl["refraction_index"]->setFloat( 1.2f );

    float3 Kd = optix::make_float3( 0.0f, 1.0f - 0.0f, 1.0f );
	box_matl["transmissive_map"]->setTextureSampler( loadTexture( m_context, "", Kd ) );

	//////////////////////////////////////////////////////////////
	// Floor
	//////////////////////////////////////////////////////////////

	// Geometry
	optix::Geometry parallelogram = m_context->createGeometry();
	parallelogram->setPrimitiveCount( 1u );

	const std::string parellelogram_ptx = helpers.getPTXPath( project_name, "parallelogram.cu" );
	parallelogram->setBoundingBoxProgram( m_context->createProgramFromPTXFile( parellelogram_ptx, "bounds" ) );
	parallelogram->setIntersectionProgram( m_context->createProgramFromPTXFile( parellelogram_ptx, "intersect" ) );
	float3 anchor = optix::make_float3( -20.0f, 0.01f, 20.0f);
	float3 v1 = optix::make_float3( 40, 0, 0);
	float3 v2 = optix::make_float3( 0, 0, -40);
	float3 normal = cross( v1, v2 );
	normal = normalize( normal );
	float d = dot( normal, anchor );
	v1 *= 1.0f/dot( v1, v1 );
	v2 *= 1.0f/dot( v2, v2 );
	optix::float4 plane = optix::make_float4( normal, d );
	parallelogram["plane"]->setFloat( plane );
	parallelogram["v1"]->setFloat( v1 );
	parallelogram["v2"]->setFloat( v2 );
	parallelogram["anchor"]->setFloat( anchor );

	// Material
	optix::Material floor_matl = m_context->createMaterial();
	floor_matl->setClosestHitProgram( 0, transparent_ch );
	floor_matl->setAnyHitProgram( 1, transparent_ah );

	floor_matl["Kd"]->setFloat( optix::make_float3( 0.7f, 0.7f, 0.7f ) );
	floor_matl["Ks"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );
	floor_matl["Ka"]->setFloat( optix::make_float3( 0.05f, 0.05f, 0.05f ) );
	floor_matl["transmissive_map"]->setTextureSampler( loadTexture( m_context, "", optix::make_float3( 0.0f, 0.0f, 0.0f ) ) );
	floor_matl["phong_exp"]->setFloat(32);
	floor_matl["refraction_index"]->setFloat( 1.0f );

	// Box
	optix::GeometryInstance box_gi = m_context->createGeometryInstance();
    box_gi->setGeometry( box );
    box_gi->setMaterialCount( 1 );
    box_gi->setMaterial( 0, box_matl );

	optix::GeometryGroup box_group = m_context->createGeometryGroup();
	box_group->setChildCount( 1u );
	box_group->setChild( 0, box_gi );
	box_group->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );

	// Floor
	optix::GeometryInstance floor_gi = m_context->createGeometryInstance( parallelogram, &floor_matl, &floor_matl+1 );

	optix::GeometryGroup floor_group = m_context->createGeometryGroup();
	floor_group->setChildCount( 1u );
	floor_group->setChild( 0, floor_gi );
	floor_group->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );

	// Transform
	optix::Transform box_transform = m_context->createTransform();
	box_transform->setChild(box_group);

	// Main Group
	m_main_group = m_context->createGroup();
	m_main_group->setChildCount(2u);
	m_main_group->setChild(0, box_transform);
	m_main_group->setChild(1, floor_group);
	m_main_group->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );

	m_context["top_object"]->set( m_main_group );
	m_context["top_shadower"]->set( m_main_group );
};
/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////
// Trace

void SimpleBoxPhysx::trace( const RayGenCameraData& camera_data )
{
	// Update Camera
	m_context["eye"]->setFloat( camera_data.eye );
	m_context["U"]->setFloat( camera_data.U );
	m_context["V"]->setFloat( camera_data.V );
	m_context["W"]->setFloat( camera_data.W );

	// Get new buffer
	optix::Buffer buffer = m_context["output_buffer"]->getBuffer();
	RTsize buffer_width, buffer_height;
	buffer->getSize( buffer_width, buffer_height );
	
	if(m_animate){
		// Update PhysX	
		if (gScene) 
		{ 
		   StepPhysX(); 
		} 

		// Update Geometry
		updateGeometry();
	}

	m_context->launch( 0, 
		static_cast<unsigned int>(buffer_width),
		static_cast<unsigned int>(buffer_height)
		);
}

void SimpleBoxPhysx::StepPhysX() 
{ 
	gScene->simulate(myTimestep);        
	       
	while(!gScene->fetchResults() )     
	{
		// do something useful        
	}
} 

void SimpleBoxPhysx::updateGeometry()
{
	//Changes go here
	optix::Transform transform = m_main_group->getChild<optix::Transform>(0);
	
	physx::PxU32 nShapes = box->getNbShapes(); 
    physx::PxShape** shapes=new physx::PxShape*[nShapes];
	
	box->getShapes(shapes, nShapes);     
    while (nShapes--) 
    { 
		physx::PxShape* shape = shapes[nShapes];
		physx::PxTransform pT = physx::PxShapeExt::getGlobalPose(*shape, *box);
		physx::PxMat33 m = physx::PxMat33(pT.q );
		float mat[16];
		convertMat(m,pT.p, mat);
		transform->setMatrix(false, mat, NULL);
    } 
	delete [] shapes;

	m_main_group->getAcceleration()->markDirty();
}
/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////
// getOutputBuffer

optix::Buffer SimpleBoxPhysx::getOutputBuffer()
{
	return m_context["output_buffer"]->getBuffer();
}
/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
// keyPressed

bool SimpleBoxPhysx::keyPressed(unsigned char key, int x, int y)
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
	physx::PxMaterial* cubeMaterial = gPhysicsSDK->createMaterial(0.5f, 0.5f, 0.5f);
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


	// Create cube	 
	physx::PxReal density = 1.0f;
	physx::PxTransform transform(physx::PxVec3(0.0f, 10.0f, 0.0f), physx::PxQuat::createIdentity());
	physx::PxBoxGeometry geometry(box_size);
    
	physx::PxRigidDynamic *actor = PxCreateDynamic(*gPhysicsSDK, transform, geometry, *cubeMaterial, density);
    actor->setAngularDamping(0.75);
    actor->setLinearVelocity(physx::PxVec3(0,0,0)); 
	if (!actor)
		std::cerr << "create actor failed!" << std::endl;
	gScene->addActor(*actor);

	box = actor;
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

	// Init GLUT
	GLUTDisplay::init( argc, argv );

	for(int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if(arg == "-P" || arg == "--pbo") {
			SimpleBoxPhysx::m_useGLBuffer = true;
		} else if( arg == "-n" || arg == "--nopbo" ) {
			SimpleBoxPhysx::m_useGLBuffer = false;
		} else if( arg == "--noanimate" ) {
			SimpleBoxPhysx::m_animate = false;
		} else if( arg == "-h" || arg == "--help" ) {
			printUsageAndExit(argv[0]);
		} else {
			std::cerr << "Unknown option '" << arg << "'\n";
			printUsageAndExit(argv[0]);
		}
	}

  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

	// Start
	try{
		SimpleBoxPhysx scene;
		GLUTDisplay::run( "Simple Box Physx", &scene, GLUTDisplay::CDAnimated);
	} catch( optix::Exception& e ){
		sutilReportError( e.getErrorString().c_str() );
		exit(1);
	}

	return 0;
}