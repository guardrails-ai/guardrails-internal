import datetime
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, Generator, Iterable
from typing import List as TypedList
from typing import Optional, Tuple, Type, TypeVar, Union

from lxml import etree as ET
from typing_extensions import Self

from guardrails.stores.document import DocumentStoreBase
from guardrails.utils.casting_utils import to_float, to_int, to_string
from guardrails.validators import Validator

if TYPE_CHECKING:
    from guardrails.schema import FormatAttr

logger = logging.getLogger(__name__)


@dataclass
class FieldValidation:
    key: Any
    value: Any
    validators: TypedList[Validator]
    children: TypedList["FieldValidation"]


def verify_metadata_requirements(
    metadata: dict, datatypes: Iterable["DataType"]
) -> TypedList[str]:
    missing_keys = set()
    for datatype in datatypes:
        for validator in datatype.validators:
            for requirement in validator.required_metadata_keys:
                if requirement not in metadata:
                    missing_keys.add(requirement)
        nested_missing_keys = verify_metadata_requirements(
            metadata, vars(datatype.children).values()
        )
        missing_keys.update(nested_missing_keys)
    return list(missing_keys)


class DataType:
    rail_alias: str

    def __init__(
        self, children: Dict[str, Any], format_attr: "FormatAttr", element: ET._Element
    ) -> None:
        self._children = children
        self.format_attr = format_attr
        self.element = element

    @property
    def validators(self) -> TypedList:
        return self.format_attr.validators

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._children})"

    def __iter__(
        self,
    ) -> Generator[Tuple[Optional[str], "DataType", ET._Element], None, None]:
        """Return a tuple of (name, child_data_type, child_element) for each
        child."""
        for el_child in self.element:
            if "name" in el_child.attrib:
                name = el_child.attrib["name"]
                if isinstance(name, bytes):
                    name = name.decode()
                child_data_type: DataType = self._children[name]
                yield name, child_data_type, el_child
            else:
                assert len(self._children) == 1, "Must have exactly one child."
                yield None, list(self._children.values())[0], el_child

    def iter(
        self, element: ET._Element
    ) -> Generator[Tuple[Optional[str], "DataType", ET._Element], None, None]:
        """Iterate over the children of an element.

        Yields tuples of (name, child_data_type, child_element) for each
        child.
        """
        for el_child in element:
            if element.tag == "list":
                assert len(self._children) == 1, "Must have exactly one child."
                yield None, list(self._children.values())[0], el_child
            else:
                name = el_child.attrib["name"]
                if isinstance(name, bytes):
                    name = name.decode()
                child_data_type: DataType = self._children[name]
                yield name, child_data_type, el_child

    def from_str(self, s: str) -> str:
        """Create a DataType from a string.

        Note: ScalarTypes like int, float, bool, etc. will override this method.
        Other ScalarTypes like string, email, url, etc. will not override this
        """
        return s

    def _constructor_validation(
        self,
        key: str,
        value: Any,
    ) -> FieldValidation:
        """Creates a "FieldValidation" object for ValidatorService to run over,
        which specifies the key, value, and validators for a given field.

        Its children should be populated by its nested fields'
        FieldValidations.
        """
        return FieldValidation(
            key=key, value=value, validators=self.validators, children=[]
        )

    def collect_validation(
        self,
        key: str,
        value: Any,
        schema: Dict,
    ) -> FieldValidation:
        """Gather validators on a value."""
        value = self.from_str(value)
        return self._constructor_validation(key, value)

    def set_children(self, element: ET._Element, document_store: DocumentStoreBase):
        raise NotImplementedError("Abstract method.")

    @classmethod
    def from_xml(
        cls,
        element: ET._Element,
        document_store: DocumentStoreBase,
        strict: bool = False,
    ) -> Self:
        from guardrails.schema import FormatAttr

        # TODO: don't want to pass strict through to DataType,
        # but need to pass it to FormatAttr.from_element
        # how to handle this?
        format_attr = FormatAttr.from_element(element)
        format_attr.get_validators(document_store, strict)

        data_type = cls({}, format_attr, element)
        data_type.set_children(element, document_store)
        return data_type

    @property
    def children(self) -> SimpleNamespace:
        """Return a SimpleNamespace of the children of this DataType."""
        return SimpleNamespace(**self._children)

    def _to_request(self) -> Dict:
        datatype = {}

        if self.format_attr is not None:
            datatype["formatters"] = self.format_attr.tokens
            if self.format_attr.namespaces is not None:
                datatype["plugins"] = self.format_attr.namespaces

        if self.element is not None:
            element = {"type": self.element.tag}

            element_name = self.element.get("name")
            element_description = self.element.get("description")
            element_strict = self.element.get("strict")
            element_date_format = self.element.get("date-format")
            element_time_format = self.element.get("time-format")
            on_fail = self.element.get("on-fail")
            on_fails = []
            attr_keys = self.element.keys()
            for attr_key in attr_keys:
                if attr_key.startswith("on-fail") and attr_key != "on-fail":
                    on_fail_method = self.element.get(attr_key)
                    validator_tag = attr_key
                    validator_on_fail = {
                        "validatorTag": validator_tag,
                        "method": on_fail_method,
                    }
                    on_fails.append(validator_on_fail)
            element_model = self.element.get("model")

            if element_name is not None:
                element["name"] = element_name
            if element_description is not None:
                element["description"] = element_description
            if element_strict is not None:
                element["strict"] = element_strict
            if element_date_format is not None:
                element["dateFormat"] = element_date_format
            if element_time_format is not None:
                element["timeFormat"] = element_time_format
            if on_fail is not None:
                element["onFail"] = on_fail
            if len(on_fails) > 0:
                element["onFails"] = on_fails
            if element_model is not None:
                element["model"] = element_model

            datatype["element"] = element

        if self.children is not None:
            serialized_children = {}
            elem_type = self.element.tag if self.element is not None else None
            elem_is_list = elem_type == "list"
            child_entries = (
                self.children.__dict__.get("item", {})
                if elem_is_list
                else self.children.__dict__
            )
            if hasattr(child_entries, "_children"):
                child_entries = child_entries._children
            for child_key in child_entries:
                serialized_children[child_key] = child_entries[child_key]._to_request()
            datatype["children"] = (
                {"item": serialized_children} if elem_is_list else serialized_children
            )
        return datatype


registry: Dict[str, Type[DataType]] = {}


T = TypeVar("T", bound=Type[DataType])


# Create a decorator to register a type
def register_type(name: str):
    def decorator(cls: T) -> T:
        registry[name] = cls
        cls.rail_alias = name
        return cls

    return decorator


class ScalarType(DataType):
    def set_children(self, element: ET._Element, document_store: DocumentStoreBase):
        for _ in element:
            raise ValueError("ScalarType data type must not have any children.")


class NonScalarType(DataType):
    pass


@register_type("string")
class String(ScalarType):
    """Element tag: `<string>`"""

    def from_str(self, s: str) -> Optional[str]:
        """Create a String from a string."""
        return to_string(s)


@register_type("integer")
class Integer(ScalarType):
    """Element tag: `<integer>`"""

    def from_str(self, s: str) -> Optional[int]:
        """Create an Integer from a string."""
        return to_int(s)


@register_type("float")
class Float(ScalarType):
    """Element tag: `<float>`"""

    def from_str(self, s: str) -> Optional[float]:
        """Create a Float from a string."""
        return to_float(s)


@register_type("bool")
class Boolean(ScalarType):
    """Element tag: `<bool>`"""

    def from_str(self, s: Union[str, bool]) -> Optional[bool]:
        """Create a Boolean from a string."""
        if s is None:
            return None

        if isinstance(s, bool):
            return s

        if s.lower() == "true":
            return True
        elif s.lower() == "false":
            return False
        else:
            raise ValueError(f"Invalid boolean value: {s}")


@register_type("date")
class Date(ScalarType):
    """Element tag: `<date>`

    To configure the date format, create a date-format attribute on the
    element. E.g. `<date name="..." ... date-format="%Y-%m-%d" />`
    """

    def __init__(
        self, children: Dict[str, Any], format_attr: "FormatAttr", element: ET._Element
    ) -> None:
        self.date_format = "%Y-%m-%d"
        super().__init__(children, format_attr, element)

    def from_str(self, s: str) -> Optional[datetime.date]:
        """Create a Date from a string."""
        if s is None:
            return None

        return datetime.datetime.strptime(s, self.date_format).date()

    @classmethod
    def from_xml(
        cls,
        element: ET._Element,
        document_store: DocumentStoreBase,
        strict: bool = False,
    ) -> "Date":
        datatype = super().from_xml(element, document_store, strict)

        if "date-format" in element.attrib or "date_format" in element.attrib:
            datatype.date_format = element.attrib["date-format"]

        return datatype


@register_type("time")
class Time(ScalarType):
    """Element tag: `<time>`

    To configure the date format, create a date-format attribute on the
    element. E.g. `<time name="..." ... time-format="%H:%M:%S" />`
    """

    def __init__(
        self, children: Dict[str, Any], format_attr: "FormatAttr", element: ET._Element
    ) -> None:
        self.time_format = "%H:%M:%S"
        super().__init__(children, format_attr, element)

    def from_str(self, s: str) -> Optional[datetime.time]:
        """Create a Time from a string."""
        if s is None:
            return None

        return datetime.datetime.strptime(s, self.time_format).time()

    @classmethod
    def from_xml(
        cls,
        element: ET._Element,
        document_store: DocumentStoreBase,
        strict: bool = False,
    ) -> "Time":
        datatype = super().from_xml(element, strict)

        if "time-format" in element.attrib or "time_format" in element.attrib:
            datatype.time_format = element.attrib["time-format"]

        return datatype


@register_type("email")
class Email(ScalarType):
    """Element tag: `<email>`"""


@register_type("url")
class URL(ScalarType):
    """Element tag: `<url>`"""


@register_type("pythoncode")
class PythonCode(ScalarType):
    """Element tag: `<pythoncode>`"""


@register_type("sql")
class SQLCode(ScalarType):
    """Element tag: `<sql>`"""


@register_type("percentage")
class Percentage(ScalarType):
    """Element tag: `<percentage>`"""


@register_type("list")
class List(NonScalarType):
    """Element tag: `<list>`"""

    def collect_validation(
        self,
        key: str,
        value: Any,
        schema: Dict,
    ) -> FieldValidation:
        # Validators in the main list data type are applied to the list overall.

        validation = self._constructor_validation(key, value)

        if len(self._children) == 0:
            return validation

        item_type = list(self._children.values())[0]

        # TODO(shreya): Edge case: List of lists -- does this still work?
        for i, item in enumerate(value):
            child_validation = item_type.collect_validation(i, item, value)
            validation.children.append(child_validation)

        return validation

    def set_children(self, element: ET._Element, document_store: DocumentStoreBase):
        for idx, child in enumerate(element, start=1):
            if idx > 1:
                # Only one child is allowed in a list data type.
                # The child must be the datatype that all items in the list
                # must conform to.
                raise ValueError("List data type must have exactly one child.")
            child_data_type = registry[child.tag]
            self._children["item"] = child_data_type.from_xml(child, document_store)


@register_type("object")
class Object(NonScalarType):
    """Element tag: `<object>`"""

    def collect_validation(
        self,
        key: str,
        value: Any,
        schema: Dict,
    ) -> FieldValidation:
        # Validators in the main object data type are applied to the object overall.

        validation = self._constructor_validation(key, value)

        if len(self._children) == 0:
            return validation

        # Types of supported children
        # 1. key_type
        # 2. value_type
        # 3. List of keys that must be present

        # TODO(shreya): Implement key type and value type later

        # Check for required keys
        for child_key, child_data_type in self._children.items():
            # Value should be a dictionary
            # child_key is an expected key that the schema defined
            # child_data_type is the data type of the expected key
            child_value = value.get(child_key, None)
            child_validation = child_data_type.collect_validation(
                child_key,
                child_value,
                value,
            )
            validation.children.append(child_validation)

        return validation

    def set_children(self, element: ET._Element, document_store: DocumentStoreBase):
        for child in element:
            child_data_type = registry[child.tag]
            name = child.attrib["name"]
            if isinstance(name, bytes):
                name = name.decode()
            self._children[name] = child_data_type.from_xml(child, document_store)


@register_type("choice")
class Choice(NonScalarType):
    """Element tag: `<object>`"""

    def __init__(
        self, children: Dict[str, Any], format_attr: "FormatAttr", element: ET._Element
    ) -> None:
        super().__init__(children, format_attr, element)
        # grab `discriminator` attribute
        self.discriminator_key = element.attrib.get("discriminator", "discriminator")

    def collect_validation(
        self,
        key: str,
        value: Any,
        schema: Dict,
    ) -> FieldValidation:
        # Validate the selected choice
        discriminator_value = value[self.discriminator_key]

        validation = self._children[discriminator_value].collect_validation(
            key,
            value,
            schema,
        )

        return validation

    def set_children(self, element: ET._Element, document_store: DocumentStoreBase):
        for child in element:
            child_data_type = registry[child.tag]
            assert child_data_type == Case
            name = child.attrib["name"]
            if isinstance(name, bytes):
                name = name.decode()
            self._children[name] = child_data_type.from_xml(child, document_store)

    @property
    def validators(self) -> TypedList:
        return []


@register_type("case")
class Case(NonScalarType):
    """Element tag: `<case>`"""

    def __init__(
        self, children: Dict[str, Any], format_attr: "FormatAttr", element: ET._Element
    ) -> None:
        super().__init__(children, format_attr, element)

    def collect_validation(
        self,
        key: str,
        value: Any,
        schema: Dict,
    ) -> FieldValidation:
        # Validate the selected choice
        validation = self._constructor_validation(key, value)

        # Collect validation for all children
        for child_key, child_data_type in self._children.items():
            # Value should be a dictionary
            # child_key is an expected key that the schema defined
            # child_data_type is the data type of the expected key
            child_value = value.get(child_key, None)
            child_validation = child_data_type.collect_validation(
                child_key,
                child_value,
                value,
            )
            validation.children.append(child_validation)

        return validation

    def set_children(self, element: ET._Element, document_store: DocumentStoreBase):
        for child in element:
            child_data_type = registry[child.tag]
            name = child.attrib["name"]
            if isinstance(name, bytes):
                name = name.decode()
            self._children[name] = child_data_type.from_xml(child, document_store)


# @register_type("key")
# class Key(DataType):
# """
# Element tag: `<string>`
# """


# @register_type("value")
# class Value(DataType):
# """
# Element tag: `<string>`
# """


# @register_type("item")
# class Item(DataType):
# """
# Element tag: `<string>`
# """
