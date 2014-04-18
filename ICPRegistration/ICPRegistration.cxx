#include "itkImageFileWriter.h"
#include "itkTransformFileWriter.h"

#include "itkEuler3DTransform.h"
#include "itkEuclideanDistancePointMetric.h"
#include "itkLevenbergMarquardtOptimizer.h"
#include "itkPointSetToPointSetRegistrationMethod.h"
#include "itkAffineTransform.h"

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

  typedef itk::PointSet<float, Dimension> PointSetType;
  
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
    fixedPoint[0] = -fixedPoints[fp][0];
    fixedPoint[1] = -fixedPoints[fp][1];
    fixedPoint[2] = fixedPoints[fp][2];
    fixedPointContainer->InsertElement(fp,fixedPoint);
    }
  fixedPointSet->SetPoints(fixedPointContainer);

  // Moving points
  size_t numberOfMovingPoints = movingPoints.size();
  for (size_t mp = 0; mp < numberOfMovingPoints; ++mp)
    {
    movingPoint[0] = -movingPoints[mp][0];
    movingPoint[1] = -movingPoints[mp][1];
    movingPoint[2] = movingPoints[mp][2];
    movingPointContainer->InsertElement(mp,movingPoint);
    }
  movingPointSet->SetPoints(movingPointContainer);

  // Set up the Metric
  typedef itk::EuclideanDistancePointMetric< PointSetType, PointSetType > MetricType;
  typedef MetricType::TransformType         TransformBaseType;
  typedef TransformBaseType::ParametersType ParametersType;
  typedef TransformBaseType::ScalarType     ScalarType;
  typedef TransformBaseType::JacobianType   JacobianType;
  typedef TransformBaseType::MatrixType     MatrixType;

  MetricType::Pointer metric = MetricType::New();

  // Set up a Transform
  typedef itk::Euler3DTransform<double> TransformType;
  TransformType::Pointer transform = TransformType::New();

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
  unsigned long numberOfIterations = 10000;
  double        gradientTolerance  = 1e-4; // convergence criterion
  double        valueTolerance     = 1e-4; // convergence criterion
  double        epsilonFunction    = 1e-5; // convergence criterion

  optimizer->SetScales(scales);
  optimizer->SetNumberOfIterations(numberOfIterations);
  optimizer->SetValueTolerance(valueTolerance);
  optimizer->SetGradientTolerance(gradientTolerance);
  optimizer->SetEpsilonFunction(epsilonFunction);

  // Set up registration
  transform->SetIdentity();
  registration->SetInitialTransformParameters( transform->GetParameters() );

  registration->SetMetric(metric);
  registration->SetOptimizer(optimizer);
  registration->SetTransform(transform);
  registration->SetFixedPointSet(fixedPointSet);
  registration->SetMovingPointSet(movingPointSet);

  // Connect observer
  //CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
  //optimizer->AddObserver( itk::IterationEvent(), observer );

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
  //std::cout << "Solution = " << solution << std::endl;

  // Convert to affine transform
  typedef itk::AffineTransform<double, 3> AffineTransformType;
  AffineTransformType::Pointer affine = AffineTransformType::New();
  affine->SetIdentity();
  affine->SetMatrix(transform->GetMatrix());
  affine->SetOffset(transform->GetOffset());
  
  // Use the inverse of the transform to register moving points to fixed points
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

  return EXIT_SUCCESS;
}
