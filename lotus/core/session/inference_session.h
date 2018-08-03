#pragma once

#include "core/common/common.h"
#include "core/common/profiler.h"
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/framework/framework_common.h"
#include "core/framework/ml_value.h"
#include "core/inc/op_kernel_author.h"
#include "core/platform/types.h"

namespace LotusIR {  // forward declarations
class Model;
class GraphTransformer;
class NodeArg;
}  // namespace LotusIR

namespace onnx {
class ModelProto;
}  // namespace onnx

namespace Lotus {
class IExecutionProvider;  // forward decl
class KernelDefBuilder;
class IOBinding;

class OpKernelInfo;
class OpKernel;
class CustomRegistry;

/**
  * Configuration information for a session.
  */
struct SessionOptions {
  //int num_threads; // not used now until we re-introduce threadpools for async execution
  bool enable_sequential_execution = true;  // TODO: should we default to sequential execution?

  // enable profiling for this session.
  bool enable_profiling = false;

  // the prefix of the profile file. The current time will be appended to the file name.
  std::string profile_file_prefix = "lotus_profile_";

  std::string session_logid;                       ///< logger id to use for session output
  unsigned short session_log_verbosity_level = 0;  ///< applies to session load, initialization, etc

  // enable the memory pattern optimization.
  // The idea is if the input shapes are the same, we could trace the internal memory allocation
  // and generate a memory pattern for future request. So next time we could just do one allocation
  // with a big chunk for all the internal memory allocation.
  bool enable_mem_pattern = true;

  unsigned max_num_graph_transformation_steps = 5;  // TODO choose a good default here?

  // enable the memory arena on CPU
  // Arena may pre-allocate memory for future usage.
  // set this option to false if you don't want it.
  bool enable_cpu_mem_arena = true;
};

/**
  * Pre-defined and custom metadata about the model.
  */
struct ModelMetadata {
  std::string producer_name;
  std::string graph_name;
  std::string domain;
  std::string description;
  int64_t version;
  std::unordered_map<std::string, std::string> custom_metadata_map;
};

/**
  * @brief This is the main class used to Run a model.
  * Sample simple usage:
  *  CPUExecutionProviderInfo epi;
  *  ProviderOption po{"CPUExecutionProvider", epi};
  *  SessionOptions so(vector<ProviderOption>{po});
  *  InferenceSession session_object{so};
  *  Common::Status status = session_object.Load(MODEL_URI);
  *  Common::Status status = session_object.Initialize();
  *
  *  NameMLValMap feeds;
  *  feeds.insert({});
  *  ...
  *  std::vector<std::string> output_names;
  *  output_names.insert(...);
  *  ...
  *  std::vector<MLValue> fetches;
  *  Common::Status status = session_object.Run(run_options, feeds, output_names, &fetches);
  *  process the output here...
  */

class InferenceSession {
 public:
  /**
    Create a new InferenceSession
    @param session_options Session options.
    @param logging_manager
    Optional logging manager instance that will enable per session logger output using
    session_options.session_logid as the logger id in messages.
    If nullptr, the default LoggingManager MUST have been created previously as it will be used
    for logging. This will use the default logger id in messages.
    See core/common/logging/logging.h for details on how to do that, and how LoggingManager::DefaultLogger works.
    */
  explicit InferenceSession(const SessionOptions& session_options,
                            Logging::LoggingManager* logging_manager = nullptr);

  ~InferenceSession();

  /**
    * Register an execution provider. If you've one to register, call this before invoking Initialize().
    * The order of invocation indicates the preference order as well. In other words call this method on your
    * most preferred execution provider first followed by the less preferred ones.
    * Calling this API is optional in which case Lotus will use its internal CPU execution provider.
    * @return OK if success.
    */
  Common::Status RegisterExecutionProvider(std::unique_ptr<IExecutionProvider> p_exec_provider);

  /**
    * Register a graph transformer. If you've one to register, call this before invoking Initialize().
    * Calling this API is optional.
    * @return OK if success.
    */
  Common::Status RegisterGraphTransformer(std::unique_ptr<LotusIR::GraphTransformer> p_graph_transformer);

  /**
    * Register a custom registry for operator schema and kernels.  If you've one to register, 
    * call this before invoking Initialize().
    * The order of invocation indicates the preference order as well. In other words call this method on your
    * most preferred registry first followed by the less preferred ones.
    * Calling this API is optional.
    * @return OK if success.
    */
  Common::Status RegisterCustomRegistry(std::shared_ptr<CustomRegistry> custom_registry);

  /**
    * Load an ONNX model.
    * @param model_uri absolute path of the model file.
    * @return OK if success.
    */
  Common::Status Load(const std::string& model_uri);

  /**
    * Load an ONNX model.
    * @param istream object of the model.
    * @return OK if success.
    */
  Common::Status Load(std::istream& model_istream);

  /**
    * Load an ONNX model.
    * @param protobuf object corresponding to the model file. model_proto will be copied by the API.
    * @return OK if success.
    */
  Common::Status Load(const onnx::ModelProto& model_proto);

  /**
    * Load an ONNX model.
    * @param protobuf object corresponding to the model file. This is primarily supported to support large models.
    * @return OK if success.
    */
  Common::Status Load(std::unique_ptr<onnx::ModelProto> p_model_proto);

  /**
    * Load an ONNX model.
    * @param p_model externally created Model obj. This API is here for the sake of lotus test tools only and
    * not intended for external usage.
    * @return OK if success.
    */
  Common::Status Load(std::unique_ptr<LotusIR::Model> p_model);

  /**
    * Initializes a previously loaded model. Initialization includes but is not
    * limited to graph transformations, construction of kernels, etc.
    * This method assumes that a method has been loaded previously.
    * @return OK if success
    */
  Common::Status Initialize();

  /**
    * Run a pre-loaded and pre-intialized model.
    * Multiple threads are allowed to run this function; hence its thread-safe.
    * @param feeds named inputs owned by client code and should not be changed during
    *        execution of this function.
    * @param output_names output names
    * @param p_fetches output values in the order specified by output_names.
    *        This should not be changed during execution of this function.
    * @return OK if success.
    */
  Common::Status Run(const NameMLValMap& feeds,
                     const std::vector<std::string>& output_names,
                     std::vector<MLValue>* p_fetches);

  /**
    * See Run(const NameMLValMap& feeds, const std::vector<std::string>& output_names, std::vector<MLValue>* p_fetches)
    * for details.
    * @param run_options use this to tune the Run call to your needs.
    */
  Common::Status Run(const RunOptions& run_options,
                     const NameMLValMap& feeds,
                     const std::vector<std::string>& output_names,
                     std::vector<MLValue>* p_fetches);

  /**
  * Creates a new binding object for binding inputs and outputs.
  * @param provider_type specifies the location where the inputs need to be potentially copied. See IOBinding class
  * for more info.
  */
  Common::Status NewIOBinding(LotusIR::ProviderType /*unused; preserved to not break WinML code; use below API instead*/,
                              std::unique_ptr<IOBinding>* io_binding);
  Common::Status NewIOBinding(std::unique_ptr<IOBinding>* io_binding);

  Common::Status Run(const RunOptions& run_options, IOBinding& io_binding);
  Common::Status Run(IOBinding& io_binding);

  /**
    * TEST ONLY: This API exists to facilitate testing only since today the ONNX model
    * input/outputs don't have names. Issue: https://github.com/onnx/onnx/issues/679.
    * Fetches all possible outputs of the model. The order of the outputs is as obtained
    * from Graph->GetOutputs().
    * See Run(const NameMLValMap& feeds, const std::vector<std::string>& output_names, std::vector<MLValue>* p_fetches)
    * for details.
    * @return OK if success.
    */
  Common::Status Run(const NameMLValMap& feeds,
                     std::vector<MLValue>* p_fetches);

  /**
    * @return pair.first = OK; FAIL otherwise. pair.second is non-NULL when pair.first = OK.
    * @note lifetime of the returned pointer is valid as long as the Session object is live.
    */
  std::pair<Common::Status, const ModelMetadata*> GetModelMetadata() const;

  /**
    * Get all input definitions of the model. This does not include weights. Use this
    * to get the name/type/shapes of the inputs.
    * @return pair.first = OK; FAIL otherwise. pair.second is non-NULL when pair.first = OK.
    * @note lifetime of the returned pointer is valid as long as the Session object is live.
    */
  std::pair<Common::Status, const InputDefList*> GetInputs() const;

  /**
    * Get all output definitions of the model. Use this to get the name/type/shapes of the outputs.
    * @return pair.first = OK; FAIL otherwise. pair.second is non-NULL when pair.first = OK.
    * @note lifetime of the returned pointer is valid as long as the Session object is live.
    */
  std::pair<Common::Status, const OutputDefList*> GetOutputs() const;

  /**
    * Get current num threads running Run.
    */
  int GetCurrentNumRuns();

  /**
    * Start profiling on this inference session. This simply turns on profiling events to be 
    * recorded. A corresponding EndProfiling has to follow to write profiling data to a file.
    *@param file_prefix is the prefix of the profile file. It can include a directory path. 
    */
  void StartProfiling(const std::string& file_prefix);

  /**
    * Write captured profile events in chromium format.
    @return the name of the profile file.
    */
  std::string EndProfiling();

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(InferenceSession);

  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace Lotus