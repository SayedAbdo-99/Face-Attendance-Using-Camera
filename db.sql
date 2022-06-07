
SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `attendance`
--

-- --------------------------------------------------------

--
-- Table structure for table `emp`
--

CREATE TABLE `emp` (
  `id` int(11) NOT NULL,
  `name` varchar(50) NOT NULL,
  `attDate` datetime NOT NULL,
  `delayTime` int(11) NOT NULL,
  `acc` decimal(11,0) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

--
-- Dumping data for table `emp`
--

INSERT INTO `emp` (`id`, `name`, `attDate`, `delayTime`, `acc`) VALUES
(1, 'Sayed', '2022-01-04 08:07:19', 7, '99'),
(2, 'Mohamed', '2022-01-04 09:17:19', 137, '98'),
(3, 'm', '2022-01-04 07:17:19', 7, '98'),
(5, 'Ali', '2022-01-04 11:04:17', 184, '100'),
(6, 'Sayed', '2022-01-04 11:19:39', 199, '1'),
(7, 'Sayed', '2022-01-04 11:28:02', 208, '199'),
(8, 'Sayed', '2022-01-04 11:31:56', 211, '100'),
(9, 'Sayed', '2022-01-04 11:34:43', 214, '100');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `emp`
--
ALTER TABLE `emp`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `emp`
--
ALTER TABLE `emp`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=10;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
