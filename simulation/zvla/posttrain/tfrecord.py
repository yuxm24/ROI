"""TensorFlow Example protobuf reader without the TensorFlow runtime."""
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

def tensorflow_example_class():
    """Create the small protobuf schema needed to read TFDS TFRecords."""
    file = descriptor_pb2.FileDescriptorProto(
        name="tensorflow_example.proto", package="tensorflow", syntax="proto3")

    def message(name):
        value = file.message_type.add()
        value.name = name
        return value

    def field(parent, name, number, field_type, *, repeated=False,
              type_name=None, oneof=None):
        value = parent.field.add()
        value.name = name
        value.number = number
        value.type = field_type
        value.label = 3 if repeated else 1
        if type_name is not None:
            value.type_name = type_name
        if oneof is not None:
            value.oneof_index = oneof

    bytes_list = message("BytesList")
    field(bytes_list, "value", 1, 12, repeated=True)
    float_list = message("FloatList")
    field(float_list, "value", 1, 2, repeated=True)
    int64_list = message("Int64List")
    field(int64_list, "value", 1, 3, repeated=True)

    feature = message("Feature")
    feature.oneof_decl.add().name = "kind"
    field(feature, "bytes_list", 1, 11,
          type_name=".tensorflow.BytesList", oneof=0)
    field(feature, "float_list", 2, 11,
          type_name=".tensorflow.FloatList", oneof=0)
    field(feature, "int64_list", 3, 11,
          type_name=".tensorflow.Int64List", oneof=0)

    features = message("Features")
    entry = features.nested_type.add()
    entry.name = "FeatureEntry"
    entry.options.map_entry = True
    field(entry, "key", 1, 9)
    field(entry, "value", 2, 11, type_name=".tensorflow.Feature")
    field(features, "feature", 1, 11, repeated=True,
          type_name=".tensorflow.Features.FeatureEntry")

    example = message("Example")
    field(example, "features", 1, 11, type_name=".tensorflow.Features")
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    return message_factory.GetMessageClass(
        pool.FindMessageTypeByName("tensorflow.Example"))

TF_EXAMPLE = tensorflow_example_class()
