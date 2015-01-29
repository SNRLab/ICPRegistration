#include "itkImageFileWriter.h"
#include "itkTransformFileWriter.h"

#include "itkEuler3DTransform.h"
#include "itkEuclideanDistancePointMetric.h"
#include "itkLevenbergMarquardtOptimizer.h"
#include "itkPointSetToPointSetRegistrationMethod.h"
#include "itkAffineTransform.h"
#include "itkTransformFileReader.h"
#include "itkTransformMeshFilter.h"
#include "itkMesh.h"

#include "itkPluginUtilities.h"

#include "ICPRegistrationCLP.h"

// Use an anonymous namespace to keep class types and function names
// from colliding when module is used as shared object module.  Every
// thing should be in an anonymous namespace except for the module
// entry point, e.g. main()
//
namespace
{

} // end of anonymous namespace


class CommandIterationUpdate : public itk::Command
{
public:
  typedef  CommandIterationUpdate   Self;
  typedef  itk::Command             Superclass;
  typedef itk::SmartPointer<Self>   Pointer;
  itkNewMacro( Self );

protected:
  CommandIterationUpdate() {};

public:

  typedef itk::LevenbergMarquardtOptimizer     OptimizerType;
  typedef const OptimizerType *                OptimizerPointer;

  void Execute(itk::Object *caller, const itk::EventObject & event)
  {
    Execute( (const itk::Object *)caller, event);
  }

  void Execute(const itk::Object * object, const itk::EventObject & event)
  {
    OptimizerPointer optimizer =
      dynamic_cast< OptimizerPointer >( object );

    if( ! itk::IterationEvent().CheckEvent( &event ) )
      {
      return;
      }

    std::cout << "Value = " << optimizer->GetCachedValue() << std::endl;
    std::cout << "Position = "  << optimizer->GetCachedCurrentPosition();
    std::cout << std::endl << std::endl;

  }

};


//-----------------------------------------------------------------------------
int main( int argc, char * argv[] )
{
  PARSE_ARGS;

  const unsigned int Dimension = 3;

  typedef itk::Mesh<float, Dimension> PointSetType;

  PointSetType::Pointer fixedPointSet  = PointSetType::New();
  PointSetType::Pointer movingPointSet = PointSetType::New();

  typedef PointSetType::PointType PointType;

  typedef PointSetType::PointsContainer PointsContainer;

  PointsContainer::Pointer fixedPointContainer  = PointsContainer::New();
  PointsContainer::Pointer movingPointContainer = PointsContainer::New();

  PointType fixedPoint;
  PointType movingPoint;

  // Fixed points
  size_t numberOfFixedPoints = fixedPoints.size();
  for (size_t fp = 0; fp < numberOfFixedPoints; ++fp)
    {
    fixedPoint[0] = fixedPoints[fp][0];
    fixedPoint[1] = fixedPoints[fp][1];
    fixedPoint[2] = fixedPoints[fp][2];
    fixedPointContainer->InsertElement(fp,fixedPoint);
    }
  fixedPointSet->SetPoints(fixedPointContainer);

  // Moving points
  size_t numberOfMovingPoints = movingPoints.size();
  for (size_t mp = 0; mp < numberOfMovingPoints; ++mp)
    {
    movingPoint[0] = movingPoints[mp][0];
    movingPoint[1] = movingPoints[mp][1];
    movingPoint[2] = movingPoints[mp][2];
    movingPointContainer->InsertElement(mp,movingPoint);
    }
  movingPointSet->SetPoints(movingPointContainer);

  // Set up a Transform
  typedef itk::Euler3DTransform<double> TransformType;
  TransformType::Pointer castTransform = TransformType::New();
  TransformType::Pointer transform = TransformType::New();

  // Get Initial transform
  typedef itk::TransformFileReader TransformReaderType;
  TransformReaderType::Pointer initTransform;

  if( initialTransform != "" )
    {
    initTransform = TransformReaderType::New();
    initTransform->SetFileName( initialTransform );
    try
      {
      initTransform->Update();
      }
    catch( itk::ExceptionObject & err )
      {
      std::cerr << err << std::endl;
      return EXIT_FAILURE;
      }
    }
  else
    {
    castTransform->SetIdentity();
    transform->SetIdentity();
    }

  if( initialTransform != ""
      && initTransform->GetTransformList()->size() != 0 )
    {
    TransformReaderType::TransformType::Pointer initial
      = *(initTransform->GetTransformList()->begin() );

    typedef itk::MatrixOffsetTransformBase<double, 3, 3> DoubleMatrixOffsetType;
    typedef itk::MatrixOffsetTransformBase<float, 3, 3>  FloatMatrixOffsetType;

    DoubleMatrixOffsetType::Pointer da
      = dynamic_cast<DoubleMatrixOffsetType *>(initial.GetPointer() );
    FloatMatrixOffsetType::Pointer fa
      = dynamic_cast<FloatMatrixOffsetType *>(initial.GetPointer() );

    // Need to inverse the read transform.
    // Slicer invert the transform when saving transform to ITK files.
    if( da )
      {
      vnl_svd<double> svd(da->GetMatrix().GetVnlMatrix() );

      castTransform->SetMatrix( svd.U() * vnl_transpose(svd.V() ) );
      castTransform->SetOffset( da->GetOffset() );
      }
    else if( fa )
      {
      vnl_matrix<double> t(3, 3);
      for( int i = 0; i < 3; ++i )
        {
        for( int j = 0; j < 3; ++j )
          {
          t.put(i, j, fa->GetMatrix().GetVnlMatrix().get(i, j) );
          }
        }

      vnl_svd<double> svd( t );

      castTransform->SetMatrix( svd.U() * vnl_transpose(svd.V() ) );
      castTransform->SetOffset( fa->GetOffset() );
      }
    else
      {
      std::cout << "Initial transform is an unsupported type.\n";
      }
    castTransform->GetInverse(transform);
    }

  // Set up the Metric
  typedef itk::EuclideanDistancePointMetric< PointSetType, PointSetType > MetricType;
  typedef MetricType::TransformType         TransformBaseType;
  typedef TransformBaseType::ParametersType ParametersType;

  MetricType::Pointer metric = MetricType::New();

  // Optimizer type
  typedef itk::LevenbergMarquardtOptimizer OptimizerType;
  OptimizerType::Pointer optimizer = OptimizerType::New();
  optimizer->SetUseCostFunctionGradient(false);

  // Registration method
  typedef itk::PointSetToPointSetRegistrationMethod< PointSetType, PointSetType > RegistrationType;
  RegistrationType::Pointer registration = RegistrationType::New();

  // Scale the translation components of the Transform in the Optimizer
  OptimizerType::ScalesType scales(transform->GetNumberOfParameters());
  const double translationScale = 1000.0; // dynamic range of translations
  const double rotationScale    = 1.0;    // dynamic range of rotations

  scales[0] = 1.0 / rotationScale;
  scales[1] = 1.0 / rotationScale;
  scales[2] = 1.0 / rotationScale;
  scales[3] = 1.0 / translationScale;
  scales[4] = 1.0 / translationScale;
  scales[5] = 1.0 / translationScale;

  // Set up Optimizer
  optimizer->SetScales(scales);
  optimizer->SetNumberOfIterations(iterations);
  optimizer->SetValueTolerance(valueTolerance);
  optimizer->SetGradientTolerance(gradientTolerance);
  optimizer->SetEpsilonFunction(epsilonFunction);

  // Set up registration
  registration->SetInitialTransformParameters( transform->GetParameters() );
  registration->SetMetric(metric);
  registration->SetOptimizer(optimizer);
  registration->SetTransform(transform);
  registration->SetFixedPointSet(fixedPointSet);
  registration->SetMovingPointSet(movingPointSet);

  // Connect observer
  if (debugSwitch)
    {
    CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
    optimizer->AddObserver( itk::IterationEvent(), observer );
    }

  try
    {
    registration->Update();
    }
  catch(itk::ExceptionObject & e)
    {
    std::cerr << e << std::endl;
    return EXIT_FAILURE;
    }

  ParametersType solution = transform->GetParameters();

  if (debugSwitch)
    {
    std::cout << "Solution = " << solution << std::endl;
    }

  // Convert to affine transform
  typedef itk::AffineTransform<double, 3> AffineTransformType;
  AffineTransformType::Pointer affine = AffineTransformType::New();
  affine->SetIdentity();
  affine->SetMatrix(transform->GetMatrix());
  affine->SetOffset(transform->GetOffset());

  // Use the inverse of the transform to register moving points to fixed points
  // (Slicer will invert the transform when reading the output transform file, so
  // in order to preserve the transform, it should be inverted before saving it)
  typedef itk::TransformFileWriter TransformWriterType;
  TransformWriterType::Pointer registrationWriter = TransformWriterType::New();
  registrationWriter->SetInput(affine->GetInverseTransform());
  registrationWriter->SetFileName(registrationTransform);

  try
    {
    registrationWriter->Update();
    }
  catch (itk::ExceptionObject &err)
    {
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }


  // Compute registration error
  PointSetType::Pointer movedPointSet;
  typedef itk::TransformMeshFilter<PointSetType,PointSetType,AffineTransformType> TransformPointSetFilter;
  TransformPointSetFilter::Pointer transformPointSet = TransformPointSetFilter::New();
  transformPointSet->SetTransform(affine);
  transformPointSet->SetInput(movingPointSet);
  movedPointSet = transformPointSet->GetOutput();

  try
    {
    transformPointSet->Update();
    }
  catch (itk::ExceptionObject &err)
    {
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }


  size_t numberOfMovedPoints = movedPointSet->GetNumberOfPoints();
  double averageMinDistance = 0.0;

  for (size_t i = 0; i < numberOfMovedPoints; ++i)
    {
      PointType movedPoint = movedPointSet->GetPoint(i);
      double minDistance = 999999.0;
      for (size_t j = 0; j < numberOfFixedPoints; ++j)
	{
	  double currentDistance = std::sqrt(
	    std::pow(movedPoint[0]-fixedPoints[j][0],2)+
	    std::pow(movedPoint[1]-fixedPoints[j][1],2)+
	    std::pow(movedPoint[2]-fixedPoints[j][2],2));
	  if (currentDistance < minDistance)
	    {
	      minDistance = currentDistance;
	    }
	}
      averageMinDistance += minDistance;
    }
  averageMinDistance /= numberOfMovedPoints;
  
  icpRegistrationError = averageMinDistance;

  std::ofstream rts;
  rts.open(returnParameterFile.c_str());
  rts << "icpRegistrationError = " << icpRegistrationError << std::endl;
  rts.close();


  return EXIT_SUCCESS;
}
