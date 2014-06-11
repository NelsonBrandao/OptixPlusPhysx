#include <iostream>

// Optix
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <ImageLoader.h>

// Optix Sample Helpers
#include <sutil.h>
#include <GLUTDisplay.h>

#include "commonStructs.h"
#include "Helpers.h"


Helpers helpers;

std::string project_name = "optix_box";

//------------------------------------------------------------------------------
//
// SimpleBoxPhysx definition
//
//------------------------------------------------------------------------------

class SimpleBoxPhysx : public SampleScene
{
public:
	virtual void	      initScene( InitialCameraData& camera_data );
	virtual void		  trace( const RayGenCameraData& camera_data );
	virtual optix::Buffer getOutputBuffer();

	void createGeometry();
	void updateGeometry();

	static bool m_useGLBuffer;
private:

	optix::GeometryGroup m_geometry_group;
	optix::Group m_main_group;
	float3*				 m_vertices;

	const static int WIDTH;
	const static int HEIGHT;
};

bool SimpleBoxPhysx::m_useGLBuffer = true;
const int SimpleBoxPhysx::WIDTH  = 1024;
const int SimpleBoxPhysx::HEIGHT = 1024;


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
    camera_data = InitialCameraData( optix::make_float3( 8.3f, 4.0f, -4.8f ), // eye
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

optix::Buffer SimpleBoxPhysx::getOutputBuffer()
{
	return m_context["output_buffer"]->getBuffer();
}

void SimpleBoxPhysx::trace( const RayGenCameraData& camera_data )
{
	m_context["eye"]->setFloat( camera_data.eye );
	m_context["U"]->setFloat( camera_data.U );
	m_context["V"]->setFloat( camera_data.V );
	m_context["W"]->setFloat( camera_data.W );

	optix::Buffer buffer = m_context["output_buffer"]->getBuffer();
	RTsize buffer_width, buffer_height;
	buffer->getSize( buffer_width, buffer_height );

	updateGeometry();

	m_context->launch( 0, 
		static_cast<unsigned int>(buffer_width),
		static_cast<unsigned int>(buffer_height)
		);
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

	const float minx = 0.5;
    const float minz = 0.5; 

    box["boxmin"]->setFloat( minx, 1.5f, minz );
    box["boxmax"]->setFloat( minx + 1.0f, 2.5f, minz + 1.0f );

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

void SimpleBoxPhysx::updateGeometry()
{
	//optix::GeometryInstance gi = m_geometry_group->getChild( 0 );
	
	//Changes go here
	optix::Transform transform = m_main_group->getChild<optix::Transform>(0);
	float m[16];
	transform->getMatrix(false, m, NULL);
	m[3] += 0.5f;
	//m[7] += 5.5f;
	//m[11] += 5.5f;
	transform->setMatrix(false, m, NULL);

	m_main_group->getAcceleration()->markDirty();
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
    << std::endl;
  GLUTDisplay::printUsage();

  if ( doExit ) exit(1);
}

int main( int argc, char** argv )
{
	// Init GLUT
	GLUTDisplay::init( argc, argv );

	for(int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if(arg == "-P" || arg == "--pbo") {
			SimpleBoxPhysx::m_useGLBuffer = true;
		} else if( arg == "-n" || arg == "--nopbo" ) {
			SimpleBoxPhysx::m_useGLBuffer = false;
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