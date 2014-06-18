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

std::string project_name = "CannonBall";

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
physx::PxVec3 box_size(1,1,1);
float sphere_radius = 1.0f;

const int box_grid_width = 3, box_grid_height = 3;
const int total_boxes = box_grid_height * box_grid_width;
const int total_spheres = 1;

std::vector<physx::PxRigidActor*> boxes_actors;
std::vector<physx::PxRigidActor*> spheres_actors;
optix::Group boxes_group;
optix::Group spheres_group;


//------------------------------------------------------------------------------
//
// CannonBall definition
//
//------------------------------------------------------------------------------

class CannonBall : public SampleScene
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
	static bool m_reset;
private:
	const static int WIDTH;
	const static int HEIGHT;
};

bool CannonBall::m_useGLBuffer = false;
bool CannonBall::m_animate = true;
bool CannonBall::m_reset = false;
const int CannonBall::WIDTH  = 1024;
const int CannonBall::HEIGHT = 728;

/////////////////////////////////////////////////////////////////////////////////////////////////
// InitScene

void CannonBall::initScene( InitialCameraData& camera_data )
{
	// Two Rays, on light and one shadow
	m_context->setRayTypeCount( 2 );
	m_context->setEntryPointCount( 1 );
	m_context->setStackSize( 2520 );

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

void CannonBall::createGeometry()
{
	// Material programs
	const std::string transparency_ptx = helpers.getPTXPath( project_name, "transparent.cu" );
	optix::Program transparent_ch = m_context->createProgramFromPTXFile( transparency_ptx, "closest_hit_radiance" );
	optix::Program transparent_ah = m_context->createProgramFromPTXFile( transparency_ptx, "any_hit_shadow" );

	// Box programs
	const std::string box_ptx = helpers.getPTXPath( project_name, "box.cu" );
	optix::Program box_bounds    = m_context->createProgramFromPTXFile( box_ptx, "box_bounds" );
	optix::Program box_intersect = m_context->createProgramFromPTXFile( box_ptx, "box_intersect" );

	// Box programs
	const std::string sphere_ptx = helpers.getPTXPath( project_name, "sphere.cu" );
	optix::Program sphere_bounds    = m_context->createProgramFromPTXFile( sphere_ptx, "bounds" );
	optix::Program sphere_intersect = m_context->createProgramFromPTXFile( sphere_ptx, "intersect" );

	//////////////////////////////////////////////////////////////
	// Boxes
	//////////////////////////////////////////////////////////////

	std::vector<optix::Geometry> box_geometries;
	std::vector<optix::Material> box_materials;
	for(int i = 0; i < total_boxes; ++i)
	{
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

		float3 Kd = optix::make_float3( 0.0f, 0.8f, 1.0f );
		box_matl["transmissive_map"]->setTextureSampler( loadTexture( m_context, "", Kd ) );

		box_geometries.push_back(box);
		box_materials.push_back(box_matl);
	}

	//////////////////////////////////////////////////////////////
	// spheres
	//////////////////////////////////////////////////////////////

	std::vector<optix::Geometry> sphere_geometries;
	std::vector<optix::Material> sphere_materials;
	for(int i = 0; i < total_spheres; ++i)
	{
		// Geometry
		optix::Geometry sphere = m_context->createGeometry();
		sphere->setPrimitiveCount( 1u );
		sphere->setBoundingBoxProgram( sphere_bounds );
		sphere->setIntersectionProgram( sphere_intersect );

		sphere["sphere"]->setFloat( 0.0f, 0.0f, 0.0f, sphere_radius );
		sphere["matrix_row_0"]->setFloat( 1.0f, 0.0f, 0.0f );
		sphere["matrix_row_1"]->setFloat( 0.0f, 1.0f, 0.0f );
		sphere["matrix_row_2"]->setFloat( 0.0f, 0.0f, 1.0f );

		// Material
		optix::Material sphere_matl = m_context->createMaterial();
		sphere_matl->setClosestHitProgram( 0, transparent_ch );
		sphere_matl->setAnyHitProgram( 1, transparent_ah );

		sphere_matl["Kd"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );
		sphere_matl["Ks"]->setFloat( optix::make_float3( 0.3f, 0.3f, 0.3f ) );
		sphere_matl["Ka"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );
		float3 Kd = optix::make_float3( 0.0f, 1.0f, 0.8f );
		sphere_matl["transmissive_map"]->setTextureSampler( loadTexture( m_context, "", Kd ) );
		sphere_matl["phong_exp"]->setFloat(64);
		sphere_matl["refraction_index"]->setFloat( 1.0f );

		sphere_geometries.push_back(sphere);
		sphere_materials.push_back(sphere_matl);
	}

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

	//////////////////////////////////////////////////////////////
	// Groups
	//////////////////////////////////////////////////////////////

	// Boxes Groups
	std::vector<optix::Transform> box_ts;
	for( unsigned int i = 0; i < box_geometries.size(); ++i ) {
		// GeometryInstance
		optix::GeometryInstance gi = m_context->createGeometryInstance(); 
		gi->setGeometry( box_geometries[i] );
		gi->setMaterialCount( 1 );
		gi->setMaterial( 0, box_materials[i] );

		// GeometryGroup
		optix::GeometryGroup box_group = m_context->createGeometryGroup();
		box_group->setChildCount( 1u );
		box_group->setChild( 0, gi );
		box_group->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );

		// Transform
		optix::Transform box_transform = m_context->createTransform();
		box_transform->setChild(box_group);

		box_ts.push_back(box_transform);
	}

	boxes_group = m_context->createGroup();
	boxes_group->setChildCount( static_cast<unsigned int>(box_ts.size()) );
	for ( unsigned int i = 0; i < box_ts.size(); ++i ) { 
		boxes_group->setChild( i, box_ts[i] );
	}
	boxes_group->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );

	// Spheres Groups
	std::vector<optix::Transform> sphere_ts;
	for( unsigned int i = 0; i < sphere_geometries.size(); ++i ) {
		// GeometryInstance
		optix::GeometryInstance gi = m_context->createGeometryInstance(); 
		gi->setGeometry( sphere_geometries[i] );
		gi->setMaterialCount( 1 );
		gi->setMaterial( 0, sphere_materials[i] );

		// GeometryGroup
		optix::GeometryGroup sphere_group = m_context->createGeometryGroup();
		sphere_group->setChildCount( 1u );
		sphere_group->setChild( 0, gi );
		sphere_group->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );

		// Transform
		optix::Transform sphere_transform = m_context->createTransform();
		sphere_transform->setChild(sphere_group);

		sphere_ts.push_back(sphere_transform);
	}

	spheres_group = m_context->createGroup();
	spheres_group->setChildCount( static_cast<unsigned int>(sphere_ts.size()) );
	for ( unsigned int i = 0; i < sphere_ts.size(); ++i ) { 
		spheres_group->setChild( i, sphere_ts[i] );
	}
	spheres_group->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );

	// Floor Group
	optix::GeometryInstance floor_gi = m_context->createGeometryInstance( parallelogram, &floor_matl, &floor_matl+1 );

	optix::GeometryGroup floor_group = m_context->createGeometryGroup();
	floor_group->setChildCount( 1u );
	floor_group->setChild( 0, floor_gi );
	floor_group->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );

	// Main Group
	optix::Group main_group = m_context->createGroup();
	main_group->setChildCount(3u);
	main_group->setChild(0, boxes_group);
	main_group->setChild(1, spheres_group);
	main_group->setChild(2, floor_group);
	main_group->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );

	m_context["top_object"]->set( main_group );
	m_context["top_shadower"]->set( main_group );
};
/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////
// Trace

void CannonBall::trace( const RayGenCameraData& camera_data )
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
			StepPhysX();

		// Update Geometry
		updateGeometry();
	}
	

	m_context->launch( 0, 
		static_cast<unsigned int>(buffer_width),
		static_cast<unsigned int>(buffer_height)
		);
}

void CannonBall::StepPhysX() 
{ 
	gScene->simulate(myTimestep);        
	       
	while(!gScene->fetchResults() ){}
} 

void CannonBall::updateGeometry()
{
	// Update Boxes
	if(total_boxes > 0) {
		for( unsigned int i = 0; i < total_boxes; ++i ) {
			optix::Transform transform = boxes_group->getChild<optix::Transform>(i);
			physx::PxRigidActor* box_actor = boxes_actors.at(i);

			physx::PxU32 nShapes = box_actor->getNbShapes(); 
			physx::PxShape** shapes=new physx::PxShape*[nShapes];
	
			box_actor->getShapes(shapes, nShapes);     
			while (nShapes--) 
			{ 
				physx::PxShape* shape = shapes[nShapes];
				physx::PxTransform pT = physx::PxShapeExt::getGlobalPose(*shape, *box_actor);
				physx::PxMat33 m = physx::PxMat33(pT.q );
				float mat[16];
				convertMat(m,pT.p, mat);
				transform->setMatrix(false, mat, NULL);
			} 
			delete [] shapes;
		}

		boxes_group->getAcceleration()->markDirty();
	}

	if(total_spheres > 0) {
		for( unsigned int i = 0; i < total_spheres; ++i ) {
			optix::Transform transform = spheres_group->getChild<optix::Transform>(i);
			physx::PxRigidActor* sphere_actor = spheres_actors.at(i);

			physx::PxU32 nShapes = sphere_actor->getNbShapes(); 
			physx::PxShape** shapes=new physx::PxShape*[nShapes];
	
			sphere_actor->getShapes(shapes, nShapes);     
			while (nShapes--) 
			{ 
				physx::PxShape* shape = shapes[nShapes];
				physx::PxTransform pT = physx::PxShapeExt::getGlobalPose(*shape, *sphere_actor);
				physx::PxMat33 m = physx::PxMat33(pT.q );
				float mat[16];
				convertMat(m,pT.p, mat);
				transform->setMatrix(false, mat, NULL);
			} 
			delete [] shapes;
		}

		spheres_group->getAcceleration()->markDirty();
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////
// getOutputBuffer

optix::Buffer CannonBall::getOutputBuffer()
{
	return m_context["output_buffer"]->getBuffer();
}
/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
// keyPressed

bool CannonBall::keyPressed(unsigned char key, int x, int y)
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
	physx::PxMaterial* sphereMaterial = gPhysicsSDK->createMaterial(0.6f, 0.1f, 0.6f);
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

	float gap_x = box_size.x * 2.0f;
	float gap_y = box_size.y * 2.0f;

	// Create boxes
	if(total_boxes > 0)
	{
		physx::PxReal density = 1.0f;
		physx::PxTransform boxTransform(physx::PxVec3(0.0f, 0.0f, 0.0f));
		physx::PxBoxGeometry boxGeometry(box_size);
		for(float i = -box_grid_width/2.0f; i < box_grid_width/2.0f; ++i)
		{
			for(unsigned int j = 0; j < box_grid_height; ++j)
			{
				boxTransform.p = physx::PxVec3((i * gap_x) + 1.0f, (j * gap_y) + 1.0f, 0.0f);
	    
				physx::PxRigidDynamic *boxActor = PxCreateDynamic(*gPhysicsSDK, boxTransform, boxGeometry, *cubeMaterial, density);
				if (!boxActor)
					std::cerr << "create actor failed!" << std::endl;
				boxActor->setAngularDamping(0.75f);
				boxActor->setLinearDamping(0.01f);
				boxActor->setMass(10.0f);

				gScene->addActor(*boxActor);

				boxes_actors.push_back(boxActor);
			}
		}
	}

	// Create spheres
	if(total_spheres > 0)
	{
		physx::PxReal density = 2.0f;
		physx::PxTransform sphereTransform(physx::PxVec3(0.0f, 0.0f, 0.0f));
		physx::PxSphereGeometry sphereGeometry(sphere_radius);
		
		for(unsigned int i = 0; i < total_spheres; ++i)
		{
			sphereTransform.p = physx::PxVec3(0.0f, 0.0f, -30.0f);

			physx::PxRigidDynamic *sphereActor = PxCreateDynamic(*gPhysicsSDK, sphereTransform, sphereGeometry, *sphereMaterial, density);
			if (!sphereActor)
				std::cerr << "create actor failed!" << std::endl;
			sphereActor->setAngularDamping(0.2f);
			sphereActor->setLinearDamping(0.1f);
			sphereActor->setMass(5.0f);
			sphereActor->setLinearVelocity(physx::PxVec3(1.3f, box_grid_height * 2, 60.0f)); 

			gScene->addActor(*sphereActor);

			spheres_actors.push_back(sphereActor);
		}
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
			CannonBall::m_useGLBuffer = true;
		} else if( arg == "-n" || arg == "--nopbo" ) {
			CannonBall::m_useGLBuffer = false;
		} else if( arg == "--noanimate" ) {
			CannonBall::m_animate = false;
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
		CannonBall scene;
		GLUTDisplay::run( "Simple Box Physx", &scene, GLUTDisplay::CDAnimated);
	} catch( optix::Exception& e ){
		sutilReportError( e.getErrorString().c_str() );
		exit(1);
	}

	return 0;
}