# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
from .stage import Stage


class Boost(Stage):
    """
    A stage that converts the data container to sensor data.

    This stage converts the data container to sensor data, which is nothing else than converting
    all data elements into a single large numpy array (matrix). The benefit of this is that the
    data can be processed more efficiently in the following stages. The downside is that the data
    needs to be in an uniform format, which increases the memory consumption.
    """

    def run(self) -> None:
        """Converts the data container to sensor data to boost further processing stages."""
        self.data_container.convert_to_sensor_data()
