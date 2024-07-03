import test.test_Convection as Convection
import test.test_Diffusion as Diffusion
import test.test_DiffusionConvection as DiffusionConvection
import test.test_DiluteFlow as DiluteFlow
import test.test_Outflow_Convection as OutflowConvection
import test.test_RateBC_Convection as RateBCConvection
import test.test_VariableBC as VariableBC
import test.test_Adsorption as Adsorption


success_all = True
success = Convection.run(output=False)
if not success:
    print('Convection test failed')
    success_all = False

success = Diffusion.run(output=False)
if not success:
    print('Diffusion test failed')
    success_all = False

success = DiffusionConvection.run(output=False)
if not success:
    print('DiffusionConvection test failed')
    success_all = False

success = DiluteFlow.run(output=False)
if not success:
    print('DiluteFlow test failed')
    success_all = False

success = OutflowConvection.run(output=False)
if not success:
    print('OutflowConvection test failed')
    success_all = False

success = RateBCConvection.run(output=False)
if not success:
    print('RateBCConvection test failed')
    success_all = False

success = VariableBC.run(output=False)
if not success:
    print('VariableBC test failed')
    success_all = False

success = Adsorption.run(output=True)
if not success:
    print('Adsorption test failed')
    success_all = False

if success_all:
    print('all unit tests passed')
else:
    print('unit tests failed')
